// String scalar functions; `../rust/scalar_str.rs` is the duckdb-neo side.
//
// - vs_len(s): byte length of `s`.
// - vs_upper_ascii(s): ASCII letters uppercased, other bytes unchanged.
// - vs_wrap(s, prefix, suffix): `prefix || s || suffix`.
// - vs_repeat(s, n): `s` repeated `n` times; NULL for negative `n`.
// - vs_first_word(s): the prefix of `s` up to its first space.
// - vs_count_char(s, c): occurrences of the byte `c` in `s`; an error unless `c` is one byte.
// - vs_has_prefix_any(s, prefixes): whether `s` starts with any entry of the constant comma-separated `prefixes`,
//   which is split once at bind time.
// - vs_null_if_empty(s): NULL for '' and NULL, otherwise `s`.
//
// Results are built from valid VARCHAR input, so they are placed without UTF-8 validation.
#include "registry.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace duckdb::cxx;

namespace {

// Concatenates the parts inline or into one heap allocation.
varchar_t Concat(Arena &heap, std::string_view first, std::string_view second, std::string_view third) {
	const auto size = first.size() + second.size() + third.size();
	if (size > std::numeric_limits<uint32_t>::max()) {
		throw InvalidInputException("vs_wrap: the result exceeds 4 GiB");
	}
	char inlined[varchar_t::INLINE_LENGTH];
	auto *bytes = size <= varchar_t::INLINE_LENGTH ? inlined : reinterpret_cast<char *>(heap.Allocate(size));
	auto *end = std::copy(first.begin(), first.end(), bytes);
	end = std::copy(second.begin(), second.end(), end);
	std::copy(third.begin(), third.end(), end);
	return varchar_t(bytes, static_cast<uint32_t>(size));
}

void LenExec(ScalarFunction::ExecInput &input) {
	ScalarExecutor::Execute<int64_t, varchar_t>(input,
	                                            [](varchar_t value) { return static_cast<int64_t>(value.size()); });
}

void UpperAsciiExec(ScalarFunction::ExecInput &input) {
	auto result = input.GetResult();
	auto heap = result.GetHeap();
	std::string buffer;
	ScalarExecutor::Execute<varchar_t, varchar_t>(input, [&](varchar_t value) {
		buffer.assign(value.data(), value.size());
		for (auto &byte : buffer) {
			// Branch-free, like Rust's `u8::to_ascii_uppercase`: flip the case bit of 'a'..='z'.
			byte = static_cast<char>(byte ^ (static_cast<unsigned char>(byte - 'a') < 26 ? 0x20 : 0));
		}
		return heap.AddStringUnsafe(buffer);
	});
}

void WrapExec(ScalarFunction::ExecInput &input) {
	auto result = input.GetResult();
	auto heap = result.GetHeap();
	ScalarExecutor::Execute<varchar_t, varchar_t, varchar_t, varchar_t>(
	    input, [&](varchar_t value, varchar_t prefix, varchar_t suffix) {
		    return Concat(heap, prefix.view(), value.view(), suffix.view());
	    });
}

void RepeatExec(ScalarFunction::ExecInput &input) {
	auto result = input.GetResult();
	auto heap = result.GetHeap();
	std::string buffer;
	ScalarExecutor::Execute<std::optional<varchar_t>, varchar_t, int32_t>(
	    input, [&](varchar_t value, int32_t count) -> std::optional<varchar_t> {
		    if (count < 0) {
			    return std::nullopt;
		    }
		    buffer.clear();
		    for (int32_t i = 0; i < count; i++) {
			    buffer.append(value.data(), value.size());
		    }
		    return heap.AddStringUnsafe(buffer);
	    });
}

void FirstWordExec(ScalarFunction::ExecInput &input) {
	auto result = input.GetResult();
	auto heap = result.GetHeap();
	ScalarExecutor::Execute<varchar_t, varchar_t>(input, [&](varchar_t value) {
		const auto text = value.view();
		return heap.AddStringUnsafe(text.substr(0, text.find(' ')));
	});
}

void CountCharExec(ScalarFunction::ExecInput &input) {
	ScalarExecutor::Execute<int64_t, varchar_t, varchar_t>(input, [](varchar_t value, varchar_t needle) {
		if (needle.size() != 1) {
			throw InvalidInputException("vs_count_char: c must be exactly one byte");
		}
		const auto text = value.view();
		return static_cast<int64_t>(std::count(text.begin(), text.end(), needle.data()[0]));
	});
}

struct PrefixList {
	std::vector<std::string> prefixes;

	bool operator==(const PrefixList &other) const {
		return prefixes == other.prefixes;
	}
};

void HasPrefixAnyBind(ScalarFunction::BindInput &input) {
	const auto value = input.TryGetConstantArgument(1);
	if (!value || value->IsNull()) {
		throw InvalidInputException("vs_has_prefix_any: prefixes must be a non-NULL constant");
	}
	// Short strings are inlined in the varchar_t itself, so it must outlive the view.
	const auto raw = value->Get<varchar_t>();
	const auto text = raw.view();
	PrefixList list;
	size_t start = 0;
	while (true) {
		const auto comma = text.find(',', start);
		list.prefixes.emplace_back(text.substr(start, comma - start));
		if (comma == std::string_view::npos) {
			break;
		}
		start = comma + 1;
	}
	input.SetBindData<PrefixList>(std::move(list));
}

void HasPrefixAnyExec(ScalarFunction::ExecInput &input) {
	const auto &prefixes = input.GetBindData<PrefixList>().prefixes;
	auto result = input.GetResult();
	ScalarExecutor::Execute<bool, varchar_t>(input.GetArg(0), result, input.GetRowCount(), [&](varchar_t value) {
		const auto text = value.view();
		return std::any_of(prefixes.begin(), prefixes.end(), [&](const std::string &prefix) {
			return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
		});
	});
}

void NullIfEmptyExec(ScalarFunction::ExecInput &input) {
	auto result = input.GetResult();
	auto heap = result.GetHeap();
	ScalarExecutor::Execute<std::optional<varchar_t>, std::optional<varchar_t>>(
	    input, [&](std::optional<varchar_t> value) -> std::optional<varchar_t> {
		    if (!value || value->size() == 0) {
			    return std::nullopt;
		    }
		    return heap.AddStringUnsafe(value->view());
	    });
}

void RegisterLen(Connection &conn) {
	auto &factory = conn.GetFactory();
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_len").SetExecCallback(LenExec);
	function.GetSignature()
	    .AddParameter("input", factory.ParseType("VARCHAR"))
	    .SetReturnType(factory.ParseType("BIGINT"));
	conn.Register(function);
}

void RegisterUpperAscii(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto varchar = factory.ParseType("VARCHAR");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_upper_ascii").SetExecCallback(UpperAsciiExec);
	function.GetSignature().AddParameter("input", varchar).SetReturnType(varchar);
	conn.Register(function);
}

void RegisterWrap(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto varchar = factory.ParseType("VARCHAR");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_wrap").SetExecCallback(WrapExec);
	function.GetSignature()
	    .AddParameter("s", varchar)
	    .AddParameter("prefix", varchar)
	    .AddParameter("suffix", varchar)
	    .SetReturnType(varchar);
	conn.Register(function);
}

void RegisterRepeat(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto varchar = factory.ParseType("VARCHAR");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_repeat").SetExecCallback(RepeatExec);
	function.GetSignature()
	    .AddParameter("s", varchar)
	    .AddParameter("n", factory.ParseType("INTEGER"))
	    .SetReturnType(varchar);
	conn.Register(function);
}

void RegisterFirstWord(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto varchar = factory.ParseType("VARCHAR");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_first_word").SetExecCallback(FirstWordExec);
	function.GetSignature().AddParameter("input", varchar).SetReturnType(varchar);
	conn.Register(function);
}

void RegisterCountChar(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto varchar = factory.ParseType("VARCHAR");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_count_char").SetExecCallback(CountCharExec);
	function.GetSignature()
	    .AddParameter("s", varchar)
	    .AddParameter("c", varchar)
	    .SetReturnType(factory.ParseType("BIGINT"));
	conn.Register(function);
}

void RegisterHasPrefixAny(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto varchar = factory.ParseType("VARCHAR");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_has_prefix_any").SetBindCallback(HasPrefixAnyBind).SetExecCallback(HasPrefixAnyExec);
	function.GetSignature()
	    .AddParameter("s", varchar)
	    .AddParameter("prefixes", varchar)
	    .SetReturnType(factory.ParseType("BOOLEAN"));
	conn.Register(function);
}

void RegisterNullIfEmpty(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto varchar = factory.ParseType("VARCHAR");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_null_if_empty")
	    .SetExecCallback(NullIfEmptyExec)
	    .SetNullHandling(FunctionNullHandling::SPECIAL);
	function.GetSignature().AddParameter("input", varchar).SetReturnType(varchar);
	conn.Register(function);
}

} // namespace

VERSUS_FAMILY(scalar_str) {
	RegisterLen(conn);
	RegisterUpperAscii(conn);
	RegisterWrap(conn);
	RegisterRepeat(conn);
	RegisterFirstWord(conn);
	RegisterCountChar(conn);
	RegisterHasPrefixAny(conn);
	RegisterNullIfEmpty(conn);
}
