// Casts between builtin types and two custom types, `vs_cents` (a BIGINT count of cents) and `vs_hex` (a VARCHAR of
// uppercase hex digits without leading zeros, like `to_hex`). NULL converts to NULL; a row that does not convert
// raises an error, or is NULL under `TRY_CAST`.
//
// - `DOUBLE -> vs_cents`: `round(x * 100)`, half away from zero; fails on NaN, infinities and BIGINT overflow.
// - `vs_cents -> DOUBLE`: `cents / 100`; also implicit (cost 1), so `vs_cents` binds to DOUBLE parameters.
// - `VARCHAR -> vs_cents`: parses `[-]digits[.d[d]]`; fails on anything else and on overflow.
// - `vs_cents -> VARCHAR`: formats `[-]units.cc`.
// - `BIGINT -> vs_hex`: the two's complement bits in hex.
// - `vs_hex -> BIGINT`: parses 1 to 16 hex digits of either case; fails on anything else.
// - `vs_cents -> vs_hex`: the cent count in hex.
#include "registry.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

using namespace duckdb::cxx;

namespace {

constexpr const char *CENTS_TYPE = "vs_cents";
constexpr const char *HEX_TYPE = "vs_hex";
constexpr const char HEX_DIGITS[] = "0123456789ABCDEF";

InvalidInputException ConversionError(std::string_view value, const char *target) {
	return InvalidInputException("could not convert '" + std::string(value) + "' to " + target);
}

std::optional<int64_t> DoubleToCents(double value) {
	const double scaled = std::round(value * 100.0);
	// The bounds are -2^63 and 2^63; NaN is outside every range.
	if (scaled >= -9223372036854775808.0 && scaled < 9223372036854775808.0) {
		return static_cast<int64_t>(scaled);
	}
	return std::nullopt;
}

bool IsDigit(char byte) {
	return byte >= '0' && byte <= '9';
}

std::optional<int64_t> ParseCents(std::string_view text) {
	const bool negative = !text.empty() && text[0] == '-';
	size_t position = negative ? 1 : 0;
	const size_t units_start = position;
	uint64_t units = 0;
	while (position < text.size() && IsDigit(text[position])) {
		if (__builtin_mul_overflow(units, 10, &units) ||
		    __builtin_add_overflow(units, static_cast<uint64_t>(text[position] - '0'), &units)) {
			return std::nullopt;
		}
		position++;
	}
	if (position == units_start) {
		return std::nullopt;
	}
	uint64_t fraction = 0;
	if (position < text.size()) {
		if (text[position] != '.') {
			return std::nullopt;
		}
		const auto digits = text.substr(position + 1);
		if (digits.empty() || digits.size() > 2) {
			return std::nullopt;
		}
		for (const char digit : digits) {
			if (!IsDigit(digit)) {
				return std::nullopt;
			}
		}
		for (const char digit : digits) {
			fraction = fraction * 10 + static_cast<uint64_t>(digit - '0');
		}
		if (digits.size() == 1) {
			fraction *= 10;
		}
	}
	uint64_t magnitude;
	if (__builtin_mul_overflow(units, 100, &magnitude) || __builtin_add_overflow(magnitude, fraction, &magnitude)) {
		return std::nullopt;
	}
	const uint64_t limit = negative ? uint64_t(1) << 63 : static_cast<uint64_t>(INT64_MAX);
	if (magnitude > limit) {
		return std::nullopt;
	}
	return static_cast<int64_t>(negative ? 0 - magnitude : magnitude);
}

void PushDecimal(uint64_t value, std::string &out) {
	char digits[20];
	size_t len = 0;
	do {
		digits[len++] = static_cast<char>('0' + value % 10);
		value /= 10;
	} while (value != 0);
	while (len > 0) {
		out.push_back(digits[--len]);
	}
}

void FormatCents(int64_t cents, std::string &out) {
	out.clear();
	if (cents < 0) {
		out.push_back('-');
	}
	const uint64_t magnitude = cents < 0 ? 0 - static_cast<uint64_t>(cents) : static_cast<uint64_t>(cents);
	PushDecimal(magnitude / 100, out);
	out.push_back('.');
	const uint64_t fraction = magnitude % 100;
	out.push_back(static_cast<char>('0' + fraction / 10));
	out.push_back(static_cast<char>('0' + fraction % 10));
}

void FormatHex(int64_t value, std::string &out) {
	out.clear();
	const auto bits = static_cast<uint64_t>(value);
	int shift = 60;
	while (shift > 0 && (bits >> shift) == 0) {
		shift -= 4;
	}
	while (true) {
		out.push_back(HEX_DIGITS[(bits >> shift) & 0xF]);
		if (shift == 0) {
			break;
		}
		shift -= 4;
	}
}

std::optional<int64_t> ParseHex(std::string_view text) {
	if (text.empty() || text.size() > 16) {
		return std::nullopt;
	}
	uint64_t bits = 0;
	for (const char byte : text) {
		uint64_t digit;
		if (byte >= '0' && byte <= '9') {
			digit = static_cast<uint64_t>(byte - '0');
		} else if (byte >= 'A' && byte <= 'F') {
			digit = static_cast<uint64_t>(byte - 'A' + 10);
		} else if (byte >= 'a' && byte <= 'f') {
			digit = static_cast<uint64_t>(byte - 'a' + 10);
		} else {
			return std::nullopt;
		}
		bits = (bits << 4) | digit;
	}
	return static_cast<int64_t>(bits);
}

void DoubleToCentsExec(CastFunction::ExecInput &input) {
	const auto mode = input.GetMode();
	const auto source = input.GetInput();
	const auto view = source.GetView();
	auto output = input.GetOutput();
	auto validity = output.GetValidityMutable();
	auto *result = output.GetDataMutable<int64_t>();
	for (idx_t row = 0; row < input.GetRowCount(); row++) {
		if (!view.IsValid(row)) {
			validity.SetInvalid(row);
			continue;
		}
		const auto cents = DoubleToCents(view.Data<double>()[view.SelAt(row)]);
		if (cents) {
			result[row] = *cents;
		} else if (mode == CastMode::TRY) {
			validity.SetInvalid(row);
		} else {
			throw InvalidInputException("DOUBLE value out of vs_cents range");
		}
	}
}

void CentsToDoubleExec(CastFunction::ExecInput &input) {
	const auto source = input.GetInput();
	const auto view = source.GetView();
	auto output = input.GetOutput();
	auto validity = output.GetValidityMutable();
	auto *result = output.GetDataMutable<double>();
	for (idx_t row = 0; row < input.GetRowCount(); row++) {
		if (!view.IsValid(row)) {
			validity.SetInvalid(row);
			continue;
		}
		result[row] = static_cast<double>(view.Data<int64_t>()[view.SelAt(row)]) / 100.0;
	}
}

void VarcharToCentsExec(CastFunction::ExecInput &input) {
	const auto mode = input.GetMode();
	const auto source = input.GetInput();
	const auto view = source.GetView();
	auto output = input.GetOutput();
	auto validity = output.GetValidityMutable();
	auto *result = output.GetDataMutable<int64_t>();
	for (idx_t row = 0; row < input.GetRowCount(); row++) {
		if (!view.IsValid(row)) {
			validity.SetInvalid(row);
			continue;
		}
		const auto text = view.Data<varchar_t>()[view.SelAt(row)].view();
		const auto cents = ParseCents(text);
		if (cents) {
			result[row] = *cents;
		} else if (mode == CastMode::TRY) {
			validity.SetInvalid(row);
		} else {
			throw ConversionError(text, CENTS_TYPE);
		}
	}
}

void CentsToVarcharExec(CastFunction::ExecInput &input) {
	const auto source = input.GetInput();
	const auto view = source.GetView();
	auto output = input.GetOutput();
	auto validity = output.GetValidityMutable();
	std::string text;
	for (idx_t row = 0; row < input.GetRowCount(); row++) {
		if (!view.IsValid(row)) {
			validity.SetInvalid(row);
			continue;
		}
		FormatCents(view.Data<int64_t>()[view.SelAt(row)], text);
		output.AssignString(row, text);
	}
}

// Formats a BIGINT-backed input as `vs_hex`; serves both `BIGINT -> vs_hex` and `vs_cents -> vs_hex`.
void IntegerToHexExec(CastFunction::ExecInput &input) {
	const auto source = input.GetInput();
	const auto view = source.GetView();
	auto output = input.GetOutput();
	auto validity = output.GetValidityMutable();
	std::string text;
	for (idx_t row = 0; row < input.GetRowCount(); row++) {
		if (!view.IsValid(row)) {
			validity.SetInvalid(row);
			continue;
		}
		FormatHex(view.Data<int64_t>()[view.SelAt(row)], text);
		output.AssignString(row, text);
	}
}

void HexToBigintExec(CastFunction::ExecInput &input) {
	const auto mode = input.GetMode();
	const auto source = input.GetInput();
	const auto view = source.GetView();
	auto output = input.GetOutput();
	auto validity = output.GetValidityMutable();
	auto *result = output.GetDataMutable<int64_t>();
	for (idx_t row = 0; row < input.GetRowCount(); row++) {
		if (!view.IsValid(row)) {
			validity.SetInvalid(row);
			continue;
		}
		const auto text = view.Data<varchar_t>()[view.SelAt(row)].view();
		const auto value = ParseHex(text);
		if (value) {
			result[row] = *value;
		} else if (mode == CastMode::TRY) {
			validity.SetInvalid(row);
		} else {
			throw ConversionError(text, "BIGINT");
		}
	}
}

void RegisterCast(Connection &conn, const LogicalType &source, const LogicalType &target,
                  CastFunction::ExecCallback callback, int64_t implicit_cast_cost = -1) {
	auto cast = CastFunction::Create(conn.GetFactory());
	cast.SetSourceType(source).SetTargetType(target).SetImplicitCastCost(implicit_cast_cost).SetExecCallback(callback);
	conn.Register(cast);
}

} // namespace

VERSUS_FAMILY(cast) {
	auto &factory = conn.GetFactory();
	const auto bigint = factory.ParseType("BIGINT");
	const auto dbl = factory.ParseType("DOUBLE");
	const auto varchar = factory.ParseType("VARCHAR");
	auto cents_type = CustomType::Create(factory);
	cents_type.SetName(CENTS_TYPE).SetBaseType(factory.ParseType("BIGINT"));
	conn.Register(cents_type);
	auto hex_type = CustomType::Create(factory);
	hex_type.SetName(HEX_TYPE).SetBaseType(factory.ParseType("VARCHAR"));
	conn.Register(hex_type);
	const auto cents = bigint.WithAlias(factory, CENTS_TYPE);
	const auto hex = varchar.WithAlias(factory, HEX_TYPE);

	RegisterCast(conn, dbl, cents, DoubleToCentsExec);
	RegisterCast(conn, cents, dbl, CentsToDoubleExec, 1);
	RegisterCast(conn, varchar, cents, VarcharToCentsExec);
	RegisterCast(conn, cents, varchar, CentsToVarcharExec);
	RegisterCast(conn, bigint, hex, IntegerToHexExec);
	RegisterCast(conn, hex, bigint, HexToBigintExec);
	RegisterCast(conn, cents, hex, IntegerToHexExec);
}
