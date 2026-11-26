// C++ ports of the existing duckdb-neo benchmark functions in `benches/support` (specified in
// `benches/instructions.md`); the Rust side registers those modules unchanged.
//
// merge_struct(i INTEGER, s VARCHAR, m MAP(VARCHAR, INTEGER), u UNION(num INTEGER, str VARCHAR), d DECIMAL(18, 3))
//     -> STRUCT of the arguments, each NULL argument a NULL field; special NULL handling.
// dash_split(input VARCHAR, n BIGINT) -> TABLE(idx BIGINT, part VARCHAR): `input` split on '-' into at most `n`
//     parts, like Rust's `str::splitn`; NULL arguments give no rows, a negative `n` is an error.
// row_hash(s VARCHAR, d DOUBLE, b BOOLEAN) -> UBIGINT: wrapping sum of a per-row FNV-1a/splitmix64 hash, NULL over
//     no rows.
// int_sum(v BIGINT) -> BIGINT: wrapping sum of the non-NULL values, NULL without any.
// base56: a COPY format for BOOLEAN, INTEGER, BIGINT, DOUBLE and VARCHAR columns, both directions.

#include "registry.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using namespace duckdb::cxx;

namespace {

// Flattens the layouts a view cannot read.
Vector Readable(Vector vector) {
	if (vector.GetVectorType() == VectorType::OTHER) {
		vector.Flatten();
	}
	return vector;
}

// The validity of an output vector, fetched on the first NULL so that all-valid output allocates no mask.
class OutputValidity {
public:
	explicit OutputValidity(Vector &vector) : vector(vector) {
	}

	void SetValid(idx_t row) {
		if (mask.words) {
			mask.SetValid(row);
		}
	}

	void SetNull(idx_t row) {
		if (!mask.words) {
			mask = vector.GetValidityMutable();
		}
		mask.SetInvalid(row);
	}

private:
	Vector &vector;
	ValidityMask mask {nullptr};
};

//----------------------------------------------------------------------------------------------------------------------
// merge_struct
//----------------------------------------------------------------------------------------------------------------------

constexpr const char *MAP_TYPE = "MAP(VARCHAR, INTEGER)";
constexpr const char *UNION_TYPE = "UNION(num INTEGER, str VARCHAR)";
using Amount = decimal_t<18, 3>;

template <class T>
void CopyColumn(const Vector &source, Vector &target, idx_t rows) {
	const auto input = source.GetView();
	auto *data = target.GetDataMutable<T>();
	OutputValidity validity(target);
	for (idx_t row = 0; row < rows; row++) {
		const auto entry = input.SelAt(row);
		if (input.RowIsValid(entry)) {
			data[row] = input.Data<T>()[entry];
			validity.SetValid(row);
		} else {
			validity.SetNull(row);
		}
	}
}

void CopyStrings(const Vector &source, Vector &target, idx_t rows) {
	const auto input = source.GetView();
	auto *data = target.GetDataMutable<varchar_t>();
	auto heap = target.GetHeap();
	OutputValidity validity(target);
	for (idx_t row = 0; row < rows; row++) {
		const auto entry = input.SelAt(row);
		if (input.RowIsValid(entry)) {
			data[row] = heap.AddStringUnsafe(input.Data<varchar_t>()[entry].view());
			validity.SetValid(row);
		} else {
			validity.SetNull(row);
		}
	}
}

void CopyMaps(const Vector &source, Vector &target, idx_t rows) {
	const auto maps = source.GetView();
	const auto entries = source.GetChild(0);
	const auto keys = entries.GetChild(0).GetView();
	const auto values = entries.GetChild(1).GetView();
	idx_t total = 0;
	for (idx_t row = 0; row < rows; row++) {
		const auto entry = maps.SelAt(row);
		if (maps.RowIsValid(entry)) {
			total += maps.Data<list_entry_t>()[entry].length;
		}
	}

	auto out_entries = target.GetChild(0);
	out_entries.SetSize(total);
	auto out_keys = out_entries.GetChild(0);
	auto out_values = out_entries.GetChild(1);
	auto *lists = target.GetDataMutable<list_entry_t>();
	auto *key_data = out_keys.GetDataMutable<varchar_t>();
	auto key_heap = out_keys.GetHeap();
	auto *value_data = out_values.GetDataMutable<int32_t>();
	OutputValidity map_validity(target);
	OutputValidity value_validity(out_values);
	idx_t offset = 0;
	for (idx_t row = 0; row < rows; row++) {
		const auto entry = maps.SelAt(row);
		if (!maps.RowIsValid(entry)) {
			lists[row] = {offset, 0};
			map_validity.SetNull(row);
			continue;
		}
		const auto list = maps.Data<list_entry_t>()[entry];
		for (idx_t i = 0; i < list.length; i++) {
			const auto key = keys.SelAt(list.offset + i);
			if (!keys.RowIsValid(key)) {
				throw InvalidInputException("merge_struct got a NULL MAP key");
			}
			key_data[offset + i] = key_heap.AddStringUnsafe(keys.Data<varchar_t>()[key].view());
			const auto value = values.SelAt(list.offset + i);
			if (values.RowIsValid(value)) {
				value_data[offset + i] = values.Data<int32_t>()[value];
				value_validity.SetValid(offset + i);
			} else {
				value_validity.SetNull(offset + i);
			}
		}
		lists[row] = {offset, list.length};
		map_validity.SetValid(row);
		offset += list.length;
	}
}

void CopyUnions(const Vector &source, Vector &target, idx_t rows) {
	const auto unions = source.GetView();
	const auto tags = source.GetChild(0).GetView();
	const auto numbers = source.GetChild(1).GetView();
	const auto strings = source.GetChild(2).GetView();

	auto out_tags = target.GetChild(0);
	auto out_numbers = target.GetChild(1);
	auto out_strings = target.GetChild(2);
	auto *tag_data = out_tags.GetDataMutable<uint8_t>();
	auto *number_data = out_numbers.GetDataMutable<int32_t>();
	auto *string_data = out_strings.GetDataMutable<varchar_t>();
	auto string_heap = out_strings.GetHeap();
	OutputValidity union_validity(target);
	OutputValidity tag_validity(out_tags);
	OutputValidity number_validity(out_numbers);
	OutputValidity string_validity(out_strings);
	for (idx_t row = 0; row < rows; row++) {
		const auto entry = unions.SelAt(row);
		if (!unions.RowIsValid(entry)) {
			number_validity.SetNull(row);
			string_validity.SetNull(row);
			union_validity.SetNull(row);
			tag_validity.SetNull(row);
			continue;
		}
		const auto tag = tags.SelAt(entry);
		if (!tags.RowIsValid(tag) || tags.Data<uint8_t>()[tag] > 1) {
			throw InvalidInputException("invalid UNION tag");
		}
		if (tags.Data<uint8_t>()[tag] == 0) {
			string_validity.SetNull(row);
			const auto number = numbers.SelAt(entry);
			if (numbers.RowIsValid(number)) {
				number_data[row] = numbers.Data<int32_t>()[number];
				number_validity.SetValid(row);
			} else {
				number_validity.SetNull(row);
			}
			tag_data[row] = 0;
		} else {
			number_validity.SetNull(row);
			const auto string = strings.SelAt(entry);
			if (strings.RowIsValid(string)) {
				string_data[row] = string_heap.AddStringUnsafe(strings.Data<varchar_t>()[string].view());
				string_validity.SetValid(row);
			} else {
				string_validity.SetNull(row);
			}
			tag_data[row] = 1;
		}
		tag_validity.SetValid(row);
		union_validity.SetValid(row);
	}
}

void MergeStructExec(ScalarFunction::ExecInput &input) {
	const auto rows = input.GetRowCount();
	const auto ints = Readable(input.GetArg(0));
	const auto strings = Readable(input.GetArg(1));
	const auto maps = Readable(input.GetArg(2));
	const auto unions = Readable(input.GetArg(3));
	const auto amounts = Readable(input.GetArg(4));

	auto result = input.GetResult();
	result.SetSize(rows);
	if (result.GetView().validity) {
		result.GetValidityMutable().SetAllValid(rows);
	}
	auto ints_field = result.GetChild(0);
	CopyColumn<int32_t>(ints, ints_field, rows);
	auto strings_field = result.GetChild(1);
	CopyStrings(strings, strings_field, rows);
	auto maps_field = result.GetChild(2);
	CopyMaps(maps, maps_field, rows);
	auto unions_field = result.GetChild(3);
	CopyUnions(unions, unions_field, rows);
	auto amounts_field = result.GetChild(4);
	CopyColumn<Amount>(amounts, amounts_field, rows);
}

void RegisterMergeStruct(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto output =
	    std::string("STRUCT(i INTEGER, s VARCHAR, m ") + MAP_TYPE + ", u " + UNION_TYPE + ", d DECIMAL(18, 3))";
	auto function = ScalarFunction::Create(factory);
	function.SetName("merge_struct").SetExecCallback(MergeStructExec).SetNullHandling(FunctionNullHandling::SPECIAL);
	function.GetSignature()
	    .AddParameter("i", factory.CreateType(LogicalTypeId::INTEGER))
	    .AddParameter("s", factory.CreateType(LogicalTypeId::VARCHAR))
	    .AddParameter("m", factory.ParseType(MAP_TYPE))
	    .AddParameter("u", factory.ParseType(UNION_TYPE))
	    .AddParameter("d", factory.ParseType("DECIMAL(18, 3)"))
	    .SetReturnType(factory.ParseType(output));
	conn.Register(function);
}

//----------------------------------------------------------------------------------------------------------------------
// dash_split
//----------------------------------------------------------------------------------------------------------------------

struct SplitBind {
	std::string input;
	idx_t limit;
};

struct SplitScan {
	explicit SplitScan(bool has_parts) {
		if (has_parts) {
			cursor.emplace(0, 0);
		}
	}

	std::mutex mutex;
	// Where the next part starts and its number; empty once every part was produced.
	std::optional<std::pair<size_t, idx_t>> cursor;
};

// Byte ranges of the parts in the current chunk.
using SplitParts = std::vector<std::pair<size_t, size_t>>;

void DashSplitBind(TableFunction::BindInput &input) {
	auto constant = [&input](idx_t index) {
		auto value = input.TryGetConstantArgument(index);
		if (!value) {
			throw InvalidInputException("dash_split arguments must be constant");
		}
		return std::move(*value);
	};
	const auto text = constant(0);
	const auto n = constant(1);
	idx_t limit = 0;
	if (!n.IsNull()) {
		const auto value = n.Get<int64_t>();
		if (value < 0) {
			throw InvalidInputException("dash_split n must not be negative");
		}
		limit = static_cast<idx_t>(value);
	}
	// NULL arguments produce no rows.
	std::string split;
	if (text.IsNull()) {
		limit = 0;
	} else {
		split = std::string(text.Get<varchar_t>().view());
	}

	auto &factory = input.GetContext().GetFactory();
	input.AddResultColumn("idx", factory.CreateType(LogicalTypeId::BIGINT));
	input.AddResultColumn("part", factory.CreateType(LogicalTypeId::VARCHAR));
	const auto parts = static_cast<idx_t>(std::count(split.begin(), split.end(), '-')) + 1;
	input.SetCardinality(std::min(parts, limit), true);
	input.SetBindData<SplitBind>(SplitBind {std::move(split), limit});
}

void DashSplitInitGlobal(TableFunction::InitGlobalInput &input) {
	const auto &bind = input.GetBindData<SplitBind>();
	input.SetGlobalState<SplitScan>(bind.limit > 0);
}

void DashSplitInitLocal(TableFunction::InitLocalInput &input) {
	input.SetLocalState<SplitParts>();
}

void DashSplitExec(TableFunction::ExecInput &input) {
	const auto &bind = input.GetBindData<SplitBind>();
	auto &scan = input.GetGlobalState<SplitScan>();
	auto &parts = input.GetLocalState<SplitParts>();
	auto output = input.GetOutputChunk();
	const std::string_view text = bind.input;
	const auto capacity = output.GetCapacity();
	parts.clear();

	idx_t first_index = 0;
	{
		std::lock_guard<std::mutex> lock(scan.mutex);
		first_index = scan.cursor ? scan.cursor->second : 0;
		while (parts.size() < capacity && scan.cursor) {
			const auto [start, index] = *scan.cursor;
			const auto separator = index + 1 < bind.limit ? text.find('-', start) : std::string_view::npos;
			if (separator != std::string_view::npos) {
				parts.emplace_back(start, separator);
				scan.cursor = std::make_pair(separator + 1, index + 1);
			} else {
				parts.emplace_back(start, text.size());
				scan.cursor.reset();
			}
		}
	}

	for (idx_t position = 0; position < input.GetColumnCount(); position++) {
		auto vector = output.GetVector(position);
		vector.SetSize(parts.size());
		if (input.GetColumnIndex(position) == 0) {
			auto *data = vector.GetDataMutable<int64_t>();
			for (idx_t row = 0; row < parts.size(); row++) {
				data[row] = static_cast<int64_t>(first_index + row);
			}
		} else {
			auto *data = vector.GetDataMutable<varchar_t>();
			auto heap = vector.GetHeap();
			for (idx_t row = 0; row < parts.size(); row++) {
				const auto [start, end] = parts[row];
				data[row] = heap.AddStringUnsafe(text.substr(start, end - start));
			}
		}
	}
}

void RegisterDashSplit(Connection &conn) {
	auto &factory = conn.GetFactory();
	auto function = TableFunction::Create(factory);
	function.SetName("dash_split")
	    .SetBindCallback(DashSplitBind)
	    .SetInitGlobalCallback(DashSplitInitGlobal)
	    .SetInitLocalCallback(DashSplitInitLocal)
	    .SetExecCallback(DashSplitExec)
	    .SetProjectionPushdown(true);
	function.GetSignature()
	    .AddParameter("input", factory.CreateType(LogicalTypeId::VARCHAR))
	    .AddParameter("n", factory.CreateType(LogicalTypeId::BIGINT));
	conn.Register(function);
}

//----------------------------------------------------------------------------------------------------------------------
// row_hash
//----------------------------------------------------------------------------------------------------------------------

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr uint64_t FNV_PRIME = 0x00000100000001b3ULL;

uint64_t Fnv(uint64_t hash, uint8_t byte) {
	return (hash ^ byte) * FNV_PRIME;
}

uint64_t Fnv(uint64_t hash, std::string_view bytes) {
	for (const auto byte : bytes) {
		hash = Fnv(hash, static_cast<uint8_t>(byte));
	}
	return hash;
}

// Hashes one row; a NULL field hashes differently from every value.
uint64_t HashRow(std::optional<std::string_view> s, std::optional<double> d, std::optional<bool> b) {
	auto hash = FNV_OFFSET;
	hash = s ? Fnv(Fnv(hash, 1), *s) : Fnv(hash, 0);
	if (d) {
		uint64_t bits;
		std::memcpy(&bits, &*d, sizeof(bits));
		hash = Fnv(hash, 1);
		for (int byte = 0; byte < 8; byte++) {
			hash = Fnv(hash, static_cast<uint8_t>(bits >> (8 * byte)));
		}
	} else {
		hash = Fnv(hash, 0);
	}
	hash = Fnv(hash, b ? (*b ? 2 : 1) : 0);
	// The splitmix64 finalizer spreads FNV's weak low bits before rows are summed.
	hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ULL;
	hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebULL;
	return hash ^ (hash >> 31);
}

struct HashState {
	uint64_t hash = 0;
	uint64_t rows = 0;
};

void RowHashSize(AggregateFunction::SizeInput &input) {
	input.SetStateSize(sizeof(HashState));
}

void RowHashInit(AggregateFunction::InitInput &input) {
	auto states = input.GetStates();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		new (states[i]) HashState();
	}
}

void RowHashUpdate(AggregateFunction::UpdateInput &input) {
	const auto s = Readable(input.GetArg(0)).GetView();
	const auto d = Readable(input.GetArg(1)).GetView();
	const auto b = Readable(input.GetArg(2)).GetView();
	auto states = input.GetStates();
	for (idx_t row = 0; row < input.GetRowCount(); row++) {
		std::optional<std::string_view> text;
		if (s.IsValid(row)) {
			text = s.Data<varchar_t>()[s.SelAt(row)].view();
		}
		std::optional<double> number;
		if (d.IsValid(row)) {
			number = d.Data<double>()[d.SelAt(row)];
		}
		std::optional<bool> flag;
		if (b.IsValid(row)) {
			flag = b.Data<bool>()[b.SelAt(row)];
		}
		auto &state = *static_cast<HashState *>(states[row]);
		state.hash += HashRow(text, number, flag);
		state.rows++;
	}
}

void RowHashCombine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		const auto &source = *static_cast<const HashState *>(sources[i]);
		auto &target = *static_cast<HashState *>(targets[i]);
		target.hash += source.hash;
		target.rows += source.rows;
	}
}

void RowHashFinalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	auto *data = result.GetDataMutable<uint64_t>();
	OutputValidity validity(result);
	const auto offset = input.GetResultOffset();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		const auto &state = *static_cast<const HashState *>(states[i]);
		if (state.rows > 0) {
			data[offset + i] = state.hash;
			validity.SetValid(offset + i);
		} else {
			validity.SetNull(offset + i);
		}
	}
}

void RegisterRowHash(Connection &conn) {
	auto &factory = conn.GetFactory();
	auto function = AggregateFunction::Create(factory);
	function.SetName("row_hash")
	    .SetSizeCallback(RowHashSize)
	    .SetInitCallback(RowHashInit)
	    .SetUpdateCallback(RowHashUpdate)
	    .SetCombineCallback(RowHashCombine)
	    .SetFinalizeCallback(RowHashFinalize)
	    .SetOrderDependence(AggregateFunction::OrderDependence::INDEPENDENT);
	function.GetSignature()
	    .AddParameter("s", factory.CreateType(LogicalTypeId::VARCHAR))
	    .AddParameter("d", factory.CreateType(LogicalTypeId::DOUBLE))
	    .AddParameter("b", factory.CreateType(LogicalTypeId::BOOLEAN))
	    .SetReturnType(factory.CreateType(LogicalTypeId::UBIGINT));
	conn.Register(function);
}

//----------------------------------------------------------------------------------------------------------------------
// int_sum
//----------------------------------------------------------------------------------------------------------------------

struct SumState {
	int64_t sum = 0;
	uint64_t rows = 0;
};

int64_t WrappingAdd(int64_t a, int64_t b) {
	return static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
}

void IntSumSize(AggregateFunction::SizeInput &input) {
	input.SetStateSize(sizeof(SumState));
}

void IntSumInit(AggregateFunction::InitInput &input) {
	auto states = input.GetStates();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		new (states[i]) SumState();
	}
}

void IntSumUpdate(AggregateFunction::UpdateInput &input) {
	const auto values = Readable(input.GetArg(0)).GetView();
	auto states = input.GetStates();
	for (idx_t row = 0; row < input.GetRowCount(); row++) {
		const auto entry = values.SelAt(row);
		if (values.RowIsValid(entry)) {
			auto &state = *static_cast<SumState *>(states[row]);
			state.sum = WrappingAdd(state.sum, values.Data<int64_t>()[entry]);
			state.rows++;
		}
	}
}

void IntSumCombine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		const auto &source = *static_cast<const SumState *>(sources[i]);
		auto &target = *static_cast<SumState *>(targets[i]);
		target.sum = WrappingAdd(target.sum, source.sum);
		target.rows += source.rows;
	}
}

void IntSumFinalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	auto *data = result.GetDataMutable<int64_t>();
	OutputValidity validity(result);
	const auto offset = input.GetResultOffset();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		const auto &state = *static_cast<const SumState *>(states[i]);
		if (state.rows > 0) {
			data[offset + i] = state.sum;
			validity.SetValid(offset + i);
		} else {
			validity.SetNull(offset + i);
		}
	}
}

void RegisterIntSum(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto bigint = factory.CreateType(LogicalTypeId::BIGINT);
	auto function = AggregateFunction::Create(factory);
	function.SetName("int_sum")
	    .SetSizeCallback(IntSumSize)
	    .SetInitCallback(IntSumInit)
	    .SetUpdateCallback(IntSumUpdate)
	    .SetCombineCallback(IntSumCombine)
	    .SetFinalizeCallback(IntSumFinalize);
	function.GetSignature().AddParameter("v", bigint).SetReturnType(bigint);
	conn.Register(function);
}

//----------------------------------------------------------------------------------------------------------------------
// base56
//----------------------------------------------------------------------------------------------------------------------
// A file starts with a header line naming the column types, e.g. `#base56 BIGINT,VARCHAR`, followed by one line per
// row of `,`-separated fields: the base56 encoding of the value's bytes (little-endian for numbers, UTF-8 for strings),
// or `~` for NULL. Every 4 bytes encode as 6 digits, a trailing 1, 2 or 3 bytes as 2, 3 or 5 digits.

// Digits and letters without the look-alikes `0`, `1`, `I`, `O`, `l` and `o`.
constexpr char ALPHABET[] = "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz";
constexpr char NULL_FIELD = '~';
constexpr char FIELD_SEPARATOR = ',';
constexpr char ROW_SEPARATOR = '\n';
constexpr std::string_view HEADER_PREFIX = "#base56 ";

// Encoded digits for a tail of 0 to 3 bytes.
constexpr std::array<size_t, 4> TAIL_DIGITS = {0, 2, 3, 5};

constexpr std::array<uint8_t, 256> DECODE = [] {
	std::array<uint8_t, 256> table {};
	for (auto &digit : table) {
		digit = UINT8_MAX;
	}
	for (size_t digit = 0; digit < 56; digit++) {
		table[static_cast<uint8_t>(ALPHABET[digit])] = static_cast<uint8_t>(digit);
	}
	return table;
}();

uint32_t LoadLittleEndian32(const uint8_t *bytes) {
	return static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8 |
	       static_cast<uint32_t>(bytes[2]) << 16 | static_cast<uint32_t>(bytes[3]) << 24;
}

void PushDigits(uint64_t value, size_t count, std::string &out) {
	const auto start = out.size();
	out.resize(start + count);
	for (size_t index = count; index-- > 0;) {
		out[start + index] = ALPHABET[value % 56];
		value /= 56;
	}
}

// Appends the base56 encoding of `size` bytes to `out`.
void EncodeInto(const uint8_t *bytes, size_t size, std::string &out) {
	const auto blocks = size / 4;
	const auto tail = size % 4;
	out.reserve(out.size() + blocks * 6 + TAIL_DIGITS[tail]);
	for (size_t block = 0; block < blocks; block++) {
		PushDigits(LoadLittleEndian32(bytes + block * 4), 6, out);
	}
	if (tail > 0) {
		uint8_t padded[4] = {};
		std::memcpy(padded, bytes + blocks * 4, tail);
		PushDigits(LoadLittleEndian32(padded), TAIL_DIGITS[tail], out);
	}
}

uint32_t ReadDigits(std::string_view digits, size_t bytes) {
	uint64_t value = 0;
	for (const auto digit : digits) {
		const auto decoded = DECODE[static_cast<uint8_t>(digit)];
		if (decoded == UINT8_MAX) {
			throw InvalidInputException(std::string("invalid base56 digit '") + digit + "'");
		}
		value = value * 56 + decoded;
	}
	if (value >> (bytes * 8) != 0) {
		throw InvalidInputException("base56 block overflows its byte count");
	}
	return static_cast<uint32_t>(value);
}

void AppendLittleEndian(uint32_t value, size_t bytes, std::string &out) {
	for (size_t byte = 0; byte < bytes; byte++) {
		out.push_back(static_cast<char>(value >> (8 * byte)));
	}
}

// Appends the bytes encoded by `digits` to `out`.
void DecodeInto(std::string_view digits, std::string &out) {
	const auto blocks = digits.size() / 6;
	const auto tail = std::find(TAIL_DIGITS.begin(), TAIL_DIGITS.end(), digits.size() % 6);
	if (tail == TAIL_DIGITS.end()) {
		throw InvalidInputException("invalid base56 length " + std::to_string(digits.size()));
	}
	const auto tail_bytes = static_cast<size_t>(tail - TAIL_DIGITS.begin());
	out.reserve(out.size() + blocks * 4 + tail_bytes);
	for (size_t block = 0; block < blocks; block++) {
		AppendLittleEndian(ReadDigits(digits.substr(block * 6, 6), 4), 4, out);
	}
	if (tail_bytes > 0) {
		AppendLittleEndian(ReadDigits(digits.substr(blocks * 6), tail_bytes), tail_bytes, out);
	}
}

// Column types the format stores.
enum class Column : uint8_t { BOOLEAN, INTEGER, BIGINT, DOUBLE, VARCHAR };

Column ColumnFromType(const LogicalType &type) {
	switch (type.GetTypeId()) {
	case LogicalTypeId::BOOLEAN:
		return Column::BOOLEAN;
	case LogicalTypeId::INTEGER:
		return Column::INTEGER;
	case LogicalTypeId::BIGINT:
		return Column::BIGINT;
	case LogicalTypeId::DOUBLE:
		return Column::DOUBLE;
	case LogicalTypeId::VARCHAR:
		return Column::VARCHAR;
	default:
		throw InvalidInputException("base56 does not support " + type.ToText());
	}
}

const char *ColumnName(Column column) {
	switch (column) {
	case Column::BOOLEAN:
		return "BOOLEAN";
	case Column::INTEGER:
		return "INTEGER";
	case Column::BIGINT:
		return "BIGINT";
	case Column::DOUBLE:
		return "DOUBLE";
	case Column::VARCHAR:
		return "VARCHAR";
	}
	return "";
}

std::string Header(const std::vector<Column> &columns) {
	std::string header(HEADER_PREFIX);
	for (size_t index = 0; index < columns.size(); index++) {
		if (index > 0) {
			header += ',';
		}
		header += ColumnName(columns[index]);
	}
	header += ROW_SEPARATOR;
	return header;
}

struct Base56ToBind {
	std::vector<Column> columns;
	std::vector<LogicalType> types;
};

// One output file of `COPY ... TO`.
struct Base56Writer {
	explicit Base56Writer(FileHandle file) : file(std::move(file)) {
	}

	std::mutex mutex;
	FileHandle file;
};

void WriteAll(FileHandle &file, std::string_view bytes) {
	while (!bytes.empty()) {
		const auto written = file.Write(bytes.data(), bytes.size());
		if (written == 0) {
			throw InvalidInputException("base56 output file accepted no bytes");
		}
		bytes.remove_prefix(written);
	}
}

template <class T>
void EncodeLittleEndian(T value, std::string &out) {
	static_assert(sizeof(T) == 4 || sizeof(T) == 8, "fixed-width columns are 4 or 8 bytes");
	using Bits = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
	Bits bits;
	std::memcpy(&bits, &value, sizeof(bits));
	uint8_t bytes[sizeof(T)];
	for (size_t byte = 0; byte < sizeof(T); byte++) {
		bytes[byte] = static_cast<uint8_t>(bits >> (8 * byte));
	}
	EncodeInto(bytes, sizeof(T), out);
}

void EncodeField(Column column, const VectorView &view, idx_t row, std::string &out) {
	const auto entry = view.SelAt(row);
	if (!view.RowIsValid(entry)) {
		out.push_back(NULL_FIELD);
		return;
	}
	switch (column) {
	case Column::BOOLEAN: {
		const uint8_t byte = view.Data<bool>()[entry] ? 1 : 0;
		EncodeInto(&byte, 1, out);
		break;
	}
	case Column::INTEGER:
		EncodeLittleEndian(view.Data<int32_t>()[entry], out);
		break;
	case Column::BIGINT:
		EncodeLittleEndian(view.Data<int64_t>()[entry], out);
		break;
	case Column::DOUBLE:
		EncodeLittleEndian(view.Data<double>()[entry], out);
		break;
	case Column::VARCHAR: {
		const auto text = view.Data<varchar_t>()[entry];
		EncodeInto(reinterpret_cast<const uint8_t *>(text.data()), text.size(), out);
		break;
	}
	}
}

void EncodeChunk(const std::vector<Column> &columns, const DataChunk &chunk, std::string &out) {
	std::vector<VectorView> views;
	views.reserve(columns.size());
	for (idx_t index = 0; index < columns.size(); index++) {
		views.push_back(chunk.GetVector(index).GetView());
	}
	for (idx_t row = 0; row < chunk.GetRowCount(); row++) {
		for (idx_t index = 0; index < columns.size(); index++) {
			if (index > 0) {
				out.push_back(FIELD_SEPARATOR);
			}
			EncodeField(columns[index], views[index], row, out);
		}
		out.push_back(ROW_SEPARATOR);
	}
}

void Base56ToBindCallback(CopyFunction::CopyToBindInput &input) {
	Base56ToBind bind;
	for (idx_t index = 0; index < input.GetColumnCount(); index++) {
		auto type = input.GetColumnType(index);
		bind.columns.push_back(ColumnFromType(type));
		bind.types.push_back(std::move(type));
	}
	input.SetBindData<Base56ToBind>(std::move(bind));
}

void Base56ToInit(CopyFunction::CopyToInitInput &input) {
	const auto &bind = input.GetBindData<Base56ToBind>();
	const auto fs = input.GetContext().GetFileSystem();
	auto file = fs.OpenFile(input.GetFilePath(), {FileFlags::WRITE, FileFlags::FILE_CREATE_NEW});
	WriteAll(file, Header(bind.columns));
	input.SetInitData<Base56Writer>(std::move(file));
}

void Base56ToBatch(CopyFunction::CopyToBatchInput &input) {
	const auto &bind = input.GetBindData<Base56ToBind>();
	const auto batch = input.TakeBatch();
	DataChunk chunk(input.GetContext().GetFactory(), bind.types);
	auto shared = batch.CreateSharedScanState();
	auto worker = batch.CreateWorkerScanState();
	std::string encoded;
	while (batch.Scan(shared, worker, chunk)) {
		EncodeChunk(bind.columns, chunk, encoded);
	}
	input.SetBatchData<std::string>(std::move(encoded));
}

void Base56ToFlush(CopyFunction::CopyToFlushInput &input) {
	auto &writer = input.GetInitData<Base56Writer>();
	std::lock_guard<std::mutex> lock(writer.mutex);
	WriteAll(writer.file, input.GetBatchData<std::string>());
}

void Base56ToFinalize(CopyFunction::CopyToFinalizeInput &input) {
	auto &writer = input.GetInitData<Base56Writer>();
	std::lock_guard<std::mutex> lock(writer.mutex);
	writer.file.Close();
}

struct Base56FromBind {
	std::string path;
	std::vector<Column> columns;
};

// A `COPY ... FROM` read: the whole file, and the start of the first unclaimed row.
struct Base56Reader {
	Base56Reader(std::string data, size_t next_row) : data(std::move(data)), next_row(next_row) {
	}

	std::string data;
	std::atomic<size_t> next_row;
};

using FieldRange = std::pair<size_t, size_t>;

// Field byte ranges of the rows one exec call decodes, and a decoding buffer.
struct Base56Batch {
	std::vector<FieldRange> fields;
	std::string bytes;
};

size_t FindByte(const std::string &data, size_t start, size_t end, char byte) {
	const auto *found = static_cast<const char *>(std::memchr(data.data() + start, byte, end - start));
	return found ? static_cast<size_t>(found - data.data()) : end;
}

void SplitFields(const std::string &data, size_t start, size_t end, idx_t columns, std::vector<FieldRange> &fields) {
	auto line_start = start;
	while (line_start < end) {
		const auto line_end = FindByte(data, line_start, end, ROW_SEPARATOR);
		auto field_start = line_start;
		idx_t count = 0;
		while (true) {
			const auto field_end = FindByte(data, field_start, line_end, FIELD_SEPARATOR);
			fields.emplace_back(field_start, field_end);
			count++;
			if (field_end == line_end) {
				break;
			}
			field_start = field_end + 1;
		}
		if (count != columns) {
			throw InvalidInputException("base56 row has " + std::to_string(count) + " fields, expected " +
			                            std::to_string(columns));
		}
		line_start = line_end + 1;
	}
}

idx_t ClaimRows(Base56Reader &reader, idx_t capacity, idx_t columns, std::vector<FieldRange> &fields) {
	fields.clear();
	const auto &data = reader.data;
	auto start = reader.next_row.load(std::memory_order_acquire);
	while (true) {
		auto end = start;
		idx_t rows = 0;
		while (rows < capacity && end < data.size()) {
			const auto line_end = FindByte(data, end, data.size(), ROW_SEPARATOR);
			end = line_end == data.size() ? data.size() : line_end + 1;
			rows++;
		}
		// On failure another worker claimed these rows first, and `start` moves past them.
		if (reader.next_row.compare_exchange_strong(start, end, std::memory_order_acq_rel, std::memory_order_acquire)) {
			SplitFields(data, start, end, columns, fields);
			return rows;
		}
	}
}

template <class T>
T FromLittleEndian(const std::string &bytes) {
	if (bytes.size() != sizeof(T)) {
		throw InvalidInputException("base56 field has " + std::to_string(bytes.size()) + " bytes, expected " +
		                            std::to_string(sizeof(T)));
	}
	using Bits = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
	Bits bits = 0;
	for (size_t byte = 0; byte < sizeof(T); byte++) {
		bits |= static_cast<Bits>(static_cast<uint8_t>(bytes[byte])) << (8 * byte);
	}
	T value;
	std::memcpy(&value, &bits, sizeof(value));
	return value;
}

template <>
bool FromLittleEndian<bool>(const std::string &bytes) {
	if (bytes.size() == 1 && (bytes[0] == 0 || bytes[0] == 1)) {
		return bytes[0] == 1;
	}
	std::string listed;
	for (const auto byte : bytes) {
		listed += (listed.empty() ? "" : ", ") + std::to_string(static_cast<uint8_t>(byte));
	}
	throw InvalidInputException("invalid base56 BOOLEAN [" + listed + "]");
}

std::string_view FieldAt(const Base56Reader &reader, const Base56Batch &batch, idx_t row, idx_t index, idx_t columns) {
	const auto [start, end] = batch.fields[row * columns + index];
	return std::string_view(reader.data).substr(start, end - start);
}

template <class T>
void DecodeColumn(Vector &vector, idx_t index, idx_t rows, idx_t columns, const Base56Reader &reader,
                  Base56Batch &batch) {
	vector.SetSize(rows);
	auto *data = vector.GetDataMutable<T>();
	OutputValidity validity(vector);
	for (idx_t row = 0; row < rows; row++) {
		const auto field = FieldAt(reader, batch, row, index, columns);
		if (field.size() == 1 && field[0] == NULL_FIELD) {
			validity.SetNull(row);
			continue;
		}
		batch.bytes.clear();
		DecodeInto(field, batch.bytes);
		data[row] = FromLittleEndian<T>(batch.bytes);
		validity.SetValid(row);
	}
}

void DecodeStrings(Vector &vector, idx_t index, idx_t rows, idx_t columns, const Base56Reader &reader,
                   Base56Batch &batch) {
	vector.SetSize(rows);
	auto *data = vector.GetDataMutable<varchar_t>();
	auto heap = vector.GetHeap();
	OutputValidity validity(vector);
	for (idx_t row = 0; row < rows; row++) {
		const auto field = FieldAt(reader, batch, row, index, columns);
		if (field.size() == 1 && field[0] == NULL_FIELD) {
			validity.SetNull(row);
			continue;
		}
		batch.bytes.clear();
		DecodeInto(field, batch.bytes);
		// The bytes come from outside DuckDB, so `AddString` validates them.
		data[row] = heap.AddString(batch.bytes);
		validity.SetValid(row);
	}
}

void Base56FromBindCallback(CopyFunction::CopyFromBindInput &input) {
	Base56FromBind bind;
	bind.path = input.GetFilePath();
	for (idx_t index = 0; index < input.GetColumnCount(); index++) {
		bind.columns.push_back(ColumnFromType(input.GetColumnType(index)));
	}
	input.SetBindData<Base56FromBind>(std::move(bind));
}

void Base56FromInitGlobal(CopyFunction::CopyFromInitGlobalInput &input) {
	const auto &bind = input.GetBindData<Base56FromBind>();
	const auto fs = input.GetContext().GetFileSystem();
	auto file = fs.OpenFile(bind.path, {FileFlags::READ});
	std::string data(file.Size(), '\0');
	file.ReadAt(data.data(), data.size(), 0);

	const auto expected = Header(bind.columns);
	if (data.compare(0, expected.size(), expected) != 0) {
		const auto found = data.substr(0, data.find(ROW_SEPARATOR));
		throw InvalidInputException("base56 header \"" + found + "\" does not match the target columns \"" +
		                            expected.substr(0, expected.size() - 1) + "\"");
	}
	input.SetGlobalState<Base56Reader>(std::move(data), expected.size());
}

void Base56FromInitLocal(CopyFunction::CopyFromInitLocalInput &input) {
	input.SetLocalState<Base56Batch>();
}

void Base56FromExec(CopyFunction::CopyFromExecInput &input) {
	const auto &bind = input.GetBindData<Base56FromBind>();
	auto &reader = input.GetGlobalState<Base56Reader>();
	auto &batch = input.GetLocalState<Base56Batch>();
	auto output = input.GetOutputChunk();
	const auto columns = bind.columns.size();
	const auto rows = ClaimRows(reader, output.GetCapacity(), columns, batch.fields);

	for (idx_t index = 0; index < columns; index++) {
		auto vector = output.GetVector(index);
		switch (bind.columns[index]) {
		case Column::BOOLEAN:
			DecodeColumn<bool>(vector, index, rows, columns, reader, batch);
			break;
		case Column::INTEGER:
			DecodeColumn<int32_t>(vector, index, rows, columns, reader, batch);
			break;
		case Column::BIGINT:
			DecodeColumn<int64_t>(vector, index, rows, columns, reader, batch);
			break;
		case Column::DOUBLE:
			DecodeColumn<double>(vector, index, rows, columns, reader, batch);
			break;
		case Column::VARCHAR:
			DecodeStrings(vector, index, rows, columns, reader, batch);
			break;
		}
	}
}

void RegisterBase56(Connection &conn) {
	auto function = CopyFunction::Create(conn.GetFactory());
	function.SetName("base56")
	    .SetFileExtension("b56")
	    .SetCopyToBindCallback(Base56ToBindCallback)
	    .SetCopyToInitCallback(Base56ToInit)
	    .SetCopyToBatchCallback(Base56ToBatch)
	    .SetCopyToFlushCallback(Base56ToFlush)
	    .SetCopyToFinalizeCallback(Base56ToFinalize)
	    .SetCopyFromBindCallback(Base56FromBindCallback)
	    .SetCopyFromInitGlobalCallback(Base56FromInitGlobal)
	    .SetCopyFromInitLocalCallback(Base56FromInitLocal)
	    .SetCopyFromExecCallback(Base56FromExec);
	conn.Register(function);
}

} // namespace

VERSUS_FAMILY(existing) {
	RegisterMergeStruct(conn);
	RegisterDashSplit(conn);
	RegisterRowHash(conn);
	RegisterIntSum(conn);
	RegisterBase56(conn);
}
