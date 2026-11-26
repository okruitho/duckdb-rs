// Scalar functions over LIST, STRUCT, ARRAY and MAP; every function returns NULL when an argument is NULL.
//
// - vs_list_sum(BIGINT[]) -> BIGINT: wrapping sum of the non-NULL elements, 0 for an empty list.
// - vs_list_scale(l BIGINT[], k BIGINT) -> BIGINT[]: each element times k (wrapping), NULL elements stay NULL.
// - vs_range_list(n INTEGER) -> INTEGER[]: [0, 1, .., n - 1], empty for n <= 0.
// - vs_split(s VARCHAR, sep VARCHAR) -> VARCHAR[]: s split on sep, which must be a single byte.
// - vs_list_join(l VARCHAR[], sep VARCHAR) -> VARCHAR: the non-NULL elements joined by sep.
// - vs_point_norm(STRUCT(x DOUBLE, y DOUBLE)) -> DOUBLE: sqrt(x * x + y * y), NULL if a field is NULL.
// - vs_make_point(x DOUBLE, y DOUBLE) -> STRUCT(x DOUBLE, y DOUBLE).
// - vs_dot3(a DOUBLE[3], b DOUBLE[3]) -> DOUBLE: dot product, NULL if an element is NULL.
// - vs_map_sum(MAP(VARCHAR, INTEGER)) -> BIGINT: wrapping sum of the non-NULL values, 0 for an empty map.
// - vs_list_of_points(n INTEGER) -> STRUCT(x DOUBLE, y DOUBLE)[]: the points (i, i * i) for i in [0, n).
#include "registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

using namespace duckdb::cxx;

namespace {

constexpr idx_t VEC3_SIZE = 3;

// Writes NULL rows through a vector's validity mask, which is fetched and reset to all valid on first use.
class NullWriter {
public:
	NullWriter(Vector &vec, idx_t count) : vec(vec), count(count) {
	}
	void SetNull(idx_t row) {
		if (!mask.words) {
			mask = vec.GetValidityMutable();
			mask.SetAllValid(count);
		}
		mask.SetInvalid(row);
	}

private:
	Vector &vec;
	idx_t count;
	ValidityMask mask {nullptr};
};

auto WrappingAdd(int64_t a, int64_t b) -> int64_t {
	return static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
}

auto WrappingMul(int64_t a, int64_t b) -> int64_t {
	return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}

void ListSumExec(ScalarFunction::ExecInput &input) {
	const auto count = input.GetRowCount();
	const auto lists = input.GetArg(0);
	const auto elements = lists.GetChild(0);
	const auto list_view = lists.GetView();
	const auto element_view = elements.GetView();
	const auto *entries = list_view.Data<list_entry_t>();
	const auto *values = element_view.Data<int64_t>();
	auto result = input.GetResult();
	auto *sums = result.GetDataMutable<int64_t>();
	NullWriter nulls(result, count);
	for (idx_t row = 0; row < count; row++) {
		if (!list_view.IsValid(row)) {
			nulls.SetNull(row);
			continue;
		}
		const auto &entry = entries[list_view.SelAt(row)];
		int64_t sum = 0;
		for (idx_t i = entry.offset; i < entry.offset + entry.length; i++) {
			if (element_view.IsValid(i)) {
				sum = WrappingAdd(sum, values[element_view.SelAt(i)]);
			}
		}
		sums[row] = sum;
	}
}

void ListScaleExec(ScalarFunction::ExecInput &input) {
	const auto count = input.GetRowCount();
	const auto lists = input.GetArg(0);
	const auto factors = input.GetArg(1);
	const auto elements = lists.GetChild(0);
	const auto list_view = lists.GetView();
	const auto factor_view = factors.GetView();
	const auto element_view = elements.GetView();
	const auto *entries = list_view.Data<list_entry_t>();
	const auto *factor_data = factor_view.Data<int64_t>();
	const auto *values = element_view.Data<int64_t>();
	idx_t total = 0;
	for (idx_t row = 0; row < count; row++) {
		if (list_view.IsValid(row) && factor_view.IsValid(row)) {
			total += entries[list_view.SelAt(row)].length;
		}
	}
	auto result = input.GetResult();
	auto result_elements = result.GetChild(0);
	result_elements.SetSize(total);
	auto *result_entries = result.GetDataMutable<list_entry_t>();
	auto *result_values = result_elements.GetDataMutable<int64_t>();
	NullWriter row_nulls(result, count);
	NullWriter element_nulls(result_elements, total);
	idx_t offset = 0;
	for (idx_t row = 0; row < count; row++) {
		if (!list_view.IsValid(row) || !factor_view.IsValid(row)) {
			row_nulls.SetNull(row);
			continue;
		}
		const auto &entry = entries[list_view.SelAt(row)];
		const auto factor = factor_data[factor_view.SelAt(row)];
		for (idx_t i = 0; i < entry.length; i++) {
			const auto source = entry.offset + i;
			if (element_view.IsValid(source)) {
				result_values[offset + i] = WrappingMul(values[element_view.SelAt(source)], factor);
			} else {
				element_nulls.SetNull(offset + i);
			}
		}
		result_entries[row] = {offset, entry.length};
		offset += entry.length;
	}
}

void RangeListExec(ScalarFunction::ExecInput &input) {
	const auto count = input.GetRowCount();
	const auto counts = input.GetArg(0);
	const auto count_view = counts.GetView();
	const auto *count_data = count_view.Data<int32_t>();
	idx_t total = 0;
	for (idx_t row = 0; row < count; row++) {
		if (count_view.IsValid(row)) {
			total += static_cast<idx_t>(std::max(count_data[count_view.SelAt(row)], 0));
		}
	}
	auto result = input.GetResult();
	auto result_elements = result.GetChild(0);
	result_elements.SetSize(total);
	auto *result_entries = result.GetDataMutable<list_entry_t>();
	auto *result_values = result_elements.GetDataMutable<int32_t>();
	NullWriter nulls(result, count);
	idx_t offset = 0;
	for (idx_t row = 0; row < count; row++) {
		if (!count_view.IsValid(row)) {
			nulls.SetNull(row);
			continue;
		}
		const auto len = std::max(count_data[count_view.SelAt(row)], 0);
		for (int32_t value = 0; value < len; value++) {
			result_values[offset + static_cast<idx_t>(value)] = value;
		}
		result_entries[row] = {offset, static_cast<uint64_t>(len)};
		offset += static_cast<idx_t>(len);
	}
}

auto SeparatorByte(const varchar_t &separator) -> char {
	if (separator.size() != 1) {
		throw InvalidInputException("vs_split: the separator must be a single byte");
	}
	return separator.data()[0];
}

void SplitExec(ScalarFunction::ExecInput &input) {
	const auto count = input.GetRowCount();
	const auto strings = input.GetArg(0);
	const auto separators = input.GetArg(1);
	const auto text_view = strings.GetView();
	const auto separator_view = separators.GetView();
	const auto *string_data = text_view.Data<varchar_t>();
	const auto *separator_data = separator_view.Data<varchar_t>();
	idx_t total = 0;
	for (idx_t row = 0; row < count; row++) {
		if (text_view.IsValid(row) && separator_view.IsValid(row)) {
			const auto separator = SeparatorByte(separator_data[separator_view.SelAt(row)]);
			const auto string = string_data[text_view.SelAt(row)].view();
			total += static_cast<idx_t>(std::count(string.begin(), string.end(), separator)) + 1;
		}
	}
	auto result = input.GetResult();
	auto result_elements = result.GetChild(0);
	result_elements.SetSize(total);
	auto *result_entries = result.GetDataMutable<list_entry_t>();
	NullWriter nulls(result, count);
	idx_t offset = 0;
	for (idx_t row = 0; row < count; row++) {
		if (!text_view.IsValid(row) || !separator_view.IsValid(row)) {
			nulls.SetNull(row);
			continue;
		}
		const auto separator = SeparatorByte(separator_data[separator_view.SelAt(row)]);
		const auto string = string_data[text_view.SelAt(row)].view();
		idx_t len = 0;
		size_t start = 0;
		while (true) {
			const auto end = string.find(separator, start);
			// Pieces of valid UTF-8 cut at an ASCII byte are valid UTF-8.
			result_elements.AssignStringUnsafe(offset + len, string.substr(start, end - start));
			len++;
			if (end == std::string_view::npos) {
				break;
			}
			start = end + 1;
		}
		result_entries[row] = {offset, len};
		offset += len;
	}
}

void ListJoinExec(ScalarFunction::ExecInput &input) {
	const auto count = input.GetRowCount();
	const auto lists = input.GetArg(0);
	const auto separators = input.GetArg(1);
	const auto elements = lists.GetChild(0);
	const auto list_view = lists.GetView();
	const auto separator_view = separators.GetView();
	const auto element_view = elements.GetView();
	const auto *entries = list_view.Data<list_entry_t>();
	const auto *separator_data = separator_view.Data<varchar_t>();
	const auto *values = element_view.Data<varchar_t>();
	auto result = input.GetResult();
	NullWriter nulls(result, count);
	std::string joined;
	for (idx_t row = 0; row < count; row++) {
		if (!list_view.IsValid(row) || !separator_view.IsValid(row)) {
			nulls.SetNull(row);
			continue;
		}
		const auto &entry = entries[list_view.SelAt(row)];
		const auto separator = separator_data[separator_view.SelAt(row)].view();
		joined.clear();
		bool first = true;
		for (idx_t i = entry.offset; i < entry.offset + entry.length; i++) {
			if (!element_view.IsValid(i)) {
				continue;
			}
			if (!first) {
				joined += separator;
			}
			joined += values[element_view.SelAt(i)].view();
			first = false;
		}
		result.AssignStringUnsafe(row, joined);
	}
}

void PointNormExec(ScalarFunction::ExecInput &input) {
	ScalarExecutor::Execute<double, std::tuple<double, double>>(input, [](std::tuple<double, double> point) {
		const auto [x, y] = point;
		return std::sqrt(x * x + y * y);
	});
}

void MakePointExec(ScalarFunction::ExecInput &input) {
	ScalarExecutor::Execute<std::tuple<double, double>, double, double>(
	    input, [](double x, double y) { return std::make_tuple(x, y); });
}

void Dot3Exec(ScalarFunction::ExecInput &input) {
	const auto count = input.GetRowCount();
	const auto lefts = input.GetArg(0);
	const auto rights = input.GetArg(1);
	const auto left_elements = lefts.GetChild(0);
	const auto right_elements = rights.GetChild(0);
	const auto left_view = lefts.GetView();
	const auto right_view = rights.GetView();
	const auto left_element_view = left_elements.GetView();
	const auto right_element_view = right_elements.GetView();
	const auto *left_values = left_element_view.Data<double>();
	const auto *right_values = right_element_view.Data<double>();
	auto result = input.GetResult();
	auto *dots = result.GetDataMutable<double>();
	NullWriter nulls(result, count);
	for (idx_t row = 0; row < count; row++) {
		if (!left_view.IsValid(row) || !right_view.IsValid(row)) {
			nulls.SetNull(row);
			continue;
		}
		const auto left_base = left_view.SelAt(row) * VEC3_SIZE;
		const auto right_base = right_view.SelAt(row) * VEC3_SIZE;
		double sum = 0.0;
		bool valid = true;
		for (idx_t i = 0; i < VEC3_SIZE; i++) {
			const auto left = left_base + i;
			const auto right = right_base + i;
			if (!left_element_view.IsValid(left) || !right_element_view.IsValid(right)) {
				valid = false;
				break;
			}
			sum = sum + left_values[left_element_view.SelAt(left)] * right_values[right_element_view.SelAt(right)];
		}
		if (valid) {
			dots[row] = sum;
		} else {
			nulls.SetNull(row);
		}
	}
}

void MapSumExec(ScalarFunction::ExecInput &input) {
	const auto count = input.GetRowCount();
	const auto maps = input.GetArg(0);
	const auto map_entries = maps.GetChild(0);
	const auto map_values = map_entries.GetChild(1);
	const auto map_view = maps.GetView();
	const auto value_view = map_values.GetView();
	const auto *entries = map_view.Data<list_entry_t>();
	const auto *values = value_view.Data<int32_t>();
	auto result = input.GetResult();
	auto *sums = result.GetDataMutable<int64_t>();
	NullWriter nulls(result, count);
	for (idx_t row = 0; row < count; row++) {
		if (!map_view.IsValid(row)) {
			nulls.SetNull(row);
			continue;
		}
		const auto &entry = entries[map_view.SelAt(row)];
		int64_t sum = 0;
		for (idx_t i = entry.offset; i < entry.offset + entry.length; i++) {
			if (value_view.IsValid(i)) {
				sum = WrappingAdd(sum, values[value_view.SelAt(i)]);
			}
		}
		sums[row] = sum;
	}
}

void ListOfPointsExec(ScalarFunction::ExecInput &input) {
	const auto count = input.GetRowCount();
	const auto counts = input.GetArg(0);
	const auto count_view = counts.GetView();
	const auto *count_data = count_view.Data<int32_t>();
	idx_t total = 0;
	for (idx_t row = 0; row < count; row++) {
		if (count_view.IsValid(row)) {
			total += static_cast<idx_t>(std::max(count_data[count_view.SelAt(row)], 0));
		}
	}
	auto result = input.GetResult();
	auto points = result.GetChild(0);
	points.SetSize(total);
	auto xs = points.GetChild(0);
	auto ys = points.GetChild(1);
	auto *result_entries = result.GetDataMutable<list_entry_t>();
	auto *x_data = xs.GetDataMutable<double>();
	auto *y_data = ys.GetDataMutable<double>();
	NullWriter nulls(result, count);
	idx_t offset = 0;
	for (idx_t row = 0; row < count; row++) {
		if (!count_view.IsValid(row)) {
			nulls.SetNull(row);
			continue;
		}
		const auto len = std::max(count_data[count_view.SelAt(row)], 0);
		for (int32_t index = 0; index < len; index++) {
			const auto x = static_cast<double>(index);
			x_data[offset + static_cast<idx_t>(index)] = x;
			y_data[offset + static_cast<idx_t>(index)] = x * x;
		}
		result_entries[row] = {offset, static_cast<uint64_t>(len)};
		offset += static_cast<idx_t>(len);
	}
}

void Register(Connection &conn, const char *name, ScalarFunction::ExecCallback exec,
              std::initializer_list<std::pair<const char *, const char *>> parameters, const std::string &return_type) {
	auto &factory = conn.GetFactory();
	auto function = ScalarFunction::Create(factory);
	function.SetName(name).SetExecCallback(exec);
	auto signature = function.GetSignature();
	for (const auto &parameter : parameters) {
		signature.AddParameter(parameter.first, factory.ParseType(parameter.second));
	}
	signature.SetReturnType(factory.ParseType(return_type));
	conn.Register(function);
}

} // namespace

VERSUS_FAMILY(scalar_nested) {
	const auto *point = "STRUCT(x DOUBLE, y DOUBLE)";
	Register(conn, "vs_list_sum", ListSumExec, {{"input", "BIGINT[]"}}, "BIGINT");
	Register(conn, "vs_list_scale", ListScaleExec, {{"l", "BIGINT[]"}, {"k", "BIGINT"}}, "BIGINT[]");
	Register(conn, "vs_range_list", RangeListExec, {{"input", "INTEGER"}}, "INTEGER[]");
	Register(conn, "vs_split", SplitExec, {{"s", "VARCHAR"}, {"sep", "VARCHAR"}}, "VARCHAR[]");
	Register(conn, "vs_list_join", ListJoinExec, {{"l", "VARCHAR[]"}, {"sep", "VARCHAR"}}, "VARCHAR");
	Register(conn, "vs_point_norm", PointNormExec, {{"input", point}}, "DOUBLE");
	Register(conn, "vs_make_point", MakePointExec, {{"x", "DOUBLE"}, {"y", "DOUBLE"}}, point);
	Register(conn, "vs_dot3", Dot3Exec, {{"a", "DOUBLE[3]"}, {"b", "DOUBLE[3]"}}, "DOUBLE");
	Register(conn, "vs_map_sum", MapSumExec, {{"input", "MAP(VARCHAR, INTEGER)"}}, "BIGINT");
	Register(conn, "vs_list_of_points", ListOfPointsExec, {{"input", "INTEGER"}}, std::string(point) + "[]");
}
