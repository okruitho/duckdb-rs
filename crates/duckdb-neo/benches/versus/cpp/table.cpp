// Table functions, the C++ side of `rust/table.rs`. A NULL row count produces no rows, as does a negative one.
// - vs_series(n BIGINT) -> (i BIGINT): 0 .. n-1.
// - vs_grid(n BIGINT) -> (i BIGINT, sq BIGINT, half DOUBLE, tag VARCHAR, even BOOLEAN, opt INTEGER): sq = i*i
//   (wrapping), half = i / 2, tag = 'row-' || i, even = i % 2 = 0, opt NULL when i % 5 = 0 else i % 1000; projection
//   pushdown.
// - vs_words(n BIGINT, len INTEGER) -> (word VARCHAR): lpad(i::VARCHAR, len, '.') for i in 0 .. n-1; NULL len
//   produces no rows, a negative one is an error.
// - vs_repeat_row(s VARCHAR, k BIGINT, n BIGINT) -> (s VARCHAR, k BIGINT): (s, k) n times as constant vectors, NULLs
//   included; projection pushdown.
// - vs_series_step(start BIGINT, stop BIGINT, step := 1) -> (i BIGINT): range(start, stop, step); a zero step is an
//   error, any NULL argument produces no rows.
// - vs_series_filter(n BIGINT) -> (i BIGINT): vs_series that applies `i <op> constant` filters itself.
// - vs_series_par(n BIGINT) -> (i BIGINT): vs_series scanned by several threads, in no particular order.
#include "registry.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

using namespace duckdb::cxx;

namespace {

// Rows a parallel scan claims at a time, which also sizes its thread count.
constexpr int64_t PAR_BLOCK_ROWS = 2048;

auto ConstantI64(const TableFunction::BindInput &input, idx_t index) -> std::optional<int64_t> {
	auto value = input.GetConstantArgument(index);
	if (value.IsNull()) {
		return std::nullopt;
	}
	return value.Get<int64_t>();
}

auto RowCount(std::optional<int64_t> n) -> int64_t {
	return n ? std::max<int64_t>(*n, 0) : 0;
}

auto ChunkRows(const DataChunk &chunk, int64_t remaining) -> idx_t {
	return static_cast<idx_t>(std::clamp<int64_t>(remaining, 0, static_cast<int64_t>(chunk.GetCapacity())));
}

template <class T, class F>
void FillColumn(DataChunk &chunk, idx_t position, idx_t rows, F &&value) {
	auto *out = chunk.GetVector(position).GetDataMutable<T>();
	for (idx_t row = 0; row < rows; row++) {
		out[row] = value(row);
	}
}

// The first vector's size is the batch's row count, which the engine propagates to the other vectors.
void FillSeries(DataChunk &chunk, idx_t rows, int64_t first) {
	FillColumn<int64_t>(chunk, 0, rows, [&](idx_t row) { return first + static_cast<int64_t>(row); });
	chunk.GetVector(0).SetSize(rows);
}

// Writes the decimal digits of value so that they end just before end, returning where they start.
auto FormatDecimal(char *end, uint64_t value) -> char * {
	auto rest = value;
	do {
		*--end = static_cast<char>('0' + rest % 10);
		rest /= 10;
	} while (rest != 0);
	return end;
}

auto FormatTag(char (&buffer)[24], int64_t i) -> std::string_view {
	auto *end = buffer + sizeof(buffer);
	auto *start = FormatDecimal(end, static_cast<uint64_t>(i)) - 4;
	std::memcpy(start, "row-", 4);
	return std::string_view(start, static_cast<size_t>(end - start));
}

// Like lpad(i::VARCHAR, len, '.'), which keeps the leading len digits of a longer number.
void WriteWord(std::string &word, int64_t i, const std::string &padding) {
	char digits[20];
	auto *end = digits + sizeof(digits);
	auto *start = FormatDecimal(end, static_cast<uint64_t>(i));
	const auto shown = std::min(static_cast<size_t>(end - start), padding.size());
	word.assign(padding, 0, padding.size() - shown);
	word.append(start, shown);
}

// Where a single-threaded scan continues.
struct SeriesScan {
	int64_t position = 0;
};

void InitSeriesScan(TableFunction::InitLocalInput &input) {
	input.SetLocalState<SeriesScan>();
}

struct CountBind {
	int64_t count = 0;
};

auto BindSeries(TableFunction::BindInput &input) -> CountBind {
	const auto count = RowCount(ConstantI64(input, 0));
	input.AddResultColumn("i", input.GetContext().GetFactory().CreateType(LogicalTypeId::BIGINT));
	input.SetCardinality(static_cast<idx_t>(count), true);
	return CountBind {count};
}

void SeriesBind(TableFunction::BindInput &input) {
	input.SetBindData<CountBind>(BindSeries(input));
}

void SeriesExec(TableFunction::ExecInput &input) {
	const auto &bind = input.GetBindData<CountBind>();
	auto &scan = input.GetLocalState<SeriesScan>();
	auto chunk = input.GetOutputChunk();
	const auto rows = ChunkRows(chunk, bind.count - scan.position);
	FillSeries(chunk, rows, scan.position);
	scan.position += static_cast<int64_t>(rows);
}

void GridBind(TableFunction::BindInput &input) {
	const auto count = RowCount(ConstantI64(input, 0));
	auto &factory = input.GetContext().GetFactory();
	input.AddResultColumn("i", factory.CreateType(LogicalTypeId::BIGINT));
	input.AddResultColumn("sq", factory.CreateType(LogicalTypeId::BIGINT));
	input.AddResultColumn("half", factory.CreateType(LogicalTypeId::DOUBLE));
	input.AddResultColumn("tag", factory.CreateType(LogicalTypeId::VARCHAR));
	input.AddResultColumn("even", factory.CreateType(LogicalTypeId::BOOLEAN));
	input.AddResultColumn("opt", factory.CreateType(LogicalTypeId::INTEGER));
	input.SetCardinality(static_cast<idx_t>(count), true);
	input.SetBindData<CountBind>(CountBind {count});
}

void GridExec(TableFunction::ExecInput &input) {
	const auto &bind = input.GetBindData<CountBind>();
	auto &scan = input.GetLocalState<SeriesScan>();
	auto chunk = input.GetOutputChunk();
	const auto rows = ChunkRows(chunk, bind.count - scan.position);
	const auto first = scan.position;
	const auto at = [&](idx_t row) {
		return first + static_cast<int64_t>(row);
	};
	const auto column_count = input.GetColumnCount();
	for (idx_t position = 0; position < column_count; position++) {
		switch (input.GetColumnIndex(position)) {
		case 0:
			FillColumn<int64_t>(chunk, position, rows, at);
			break;
		case 1:
			FillColumn<int64_t>(chunk, position, rows, [&](idx_t row) {
				const auto i = static_cast<uint64_t>(at(row));
				return static_cast<int64_t>(i * i);
			});
			break;
		case 2:
			FillColumn<double>(chunk, position, rows, [&](idx_t row) { return static_cast<double>(at(row)) / 2.0; });
			break;
		case 3: {
			auto vec = chunk.GetVector(position);
			auto heap = vec.GetHeap();
			auto *out = vec.GetDataMutable<varchar_t>();
			char tag[24];
			for (idx_t row = 0; row < rows; row++) {
				// The text is ASCII, which Rust's `&str` guarantees on the other side.
				out[row] = heap.AddStringUnsafe(FormatTag(tag, at(row)));
			}
			break;
		}
		case 4:
			FillColumn<bool>(chunk, position, rows, [&](idx_t row) { return at(row) % 2 == 0; });
			break;
		case 5: {
			auto vec = chunk.GetVector(position);
			auto *out = vec.GetDataMutable<int32_t>();
			auto validity = vec.GetValidityMutable();
			for (idx_t row = 0; row < rows; row++) {
				const auto i = at(row);
				if (i % 5 == 0) {
					validity.SetInvalid(row);
				} else {
					out[row] = static_cast<int32_t>(i % 1000);
				}
			}
			break;
		}
		default:
			throw InvalidInputException("vs_grid has no column " + std::to_string(input.GetColumnIndex(position)));
		}
	}
	chunk.GetVector(0).SetSize(rows);
	scan.position += static_cast<int64_t>(rows);
}

// padding is len dots, the longest prefix a word needs.
struct WordsBind {
	int64_t count = 0;
	std::string padding;
};

// The reused buffer of one scan.
struct WordsScan {
	int64_t position = 0;
	std::string word;
};

void WordsBind_(TableFunction::BindInput &input) {
	auto count = RowCount(ConstantI64(input, 0));
	size_t len = 0;
	auto len_value = input.GetConstantArgument(1);
	if (len_value.IsNull()) {
		count = 0;
	} else {
		const auto requested = len_value.Get<int32_t>();
		if (requested < 0) {
			throw InvalidInputException("vs_words len must not be negative");
		}
		len = static_cast<size_t>(requested);
	}
	input.AddResultColumn("word", input.GetContext().GetFactory().CreateType(LogicalTypeId::VARCHAR));
	input.SetCardinality(static_cast<idx_t>(count), true);
	input.SetBindData<WordsBind>(WordsBind {count, std::string(len, '.')});
}

void WordsInitLocal(TableFunction::InitLocalInput &input) {
	input.SetLocalState<WordsScan>();
}

void WordsExec(TableFunction::ExecInput &input) {
	const auto &bind = input.GetBindData<WordsBind>();
	auto &scan = input.GetLocalState<WordsScan>();
	auto chunk = input.GetOutputChunk();
	const auto rows = ChunkRows(chunk, bind.count - scan.position);
	auto vec = chunk.GetVector(0);
	auto heap = vec.GetHeap();
	auto *out = vec.GetDataMutable<varchar_t>();
	for (idx_t row = 0; row < rows; row++) {
		WriteWord(scan.word, scan.position + static_cast<int64_t>(row), bind.padding);
		// The text is ASCII, which Rust's `&str` guarantees on the other side.
		out[row] = heap.AddStringUnsafe(scan.word);
	}
	vec.SetSize(rows);
	scan.position += static_cast<int64_t>(rows);
}

struct RepeatBind {
	Value s;
	Value k;
	int64_t count = 0;
};

void RepeatRowBind(TableFunction::BindInput &input) {
	auto s = input.GetConstantArgument(0);
	auto k = input.GetConstantArgument(1);
	const auto count = RowCount(ConstantI64(input, 2));
	auto &factory = input.GetContext().GetFactory();
	input.AddResultColumn("s", factory.CreateType(LogicalTypeId::VARCHAR));
	input.AddResultColumn("k", factory.CreateType(LogicalTypeId::BIGINT));
	input.SetCardinality(static_cast<idx_t>(count), true);
	input.SetBindData<RepeatBind>(RepeatBind {std::move(s), std::move(k), count});
}

void RepeatRowExec(TableFunction::ExecInput &input) {
	const auto &bind = input.GetBindData<RepeatBind>();
	auto &scan = input.GetLocalState<SeriesScan>();
	auto chunk = input.GetOutputChunk();
	const auto rows = ChunkRows(chunk, bind.count - scan.position);
	if (rows == 0) {
		return;
	}
	const auto column_count = input.GetColumnCount();
	for (idx_t position = 0; position < column_count; position++) {
		switch (input.GetColumnIndex(position)) {
		case 0:
			chunk.GetVector(position).MakeConstant(bind.s, rows);
			break;
		case 1:
			chunk.GetVector(position).MakeConstant(bind.k, rows);
			break;
		default:
			throw InvalidInputException("vs_repeat_row has no column " +
			                            std::to_string(input.GetColumnIndex(position)));
		}
	}
	scan.position += static_cast<int64_t>(rows);
}

struct StepBind {
	int64_t start = 0;
	int64_t step = 1;
	uint64_t count = 0;
};

// How many values start, start + step, ... lie before stop.
auto StepCount(int64_t start, int64_t stop, int64_t step) -> uint64_t {
	const auto ustart = static_cast<uint64_t>(start);
	const auto ustop = static_cast<uint64_t>(stop);
	if (step > 0 && start < stop) {
		return (ustop - ustart - 1) / static_cast<uint64_t>(step) + 1;
	}
	if (step < 0 && start > stop) {
		return (ustart - ustop - 1) / (uint64_t(0) - static_cast<uint64_t>(step)) + 1;
	}
	return 0;
}

struct StepScan {
	uint64_t position = 0;
};

void SeriesStepBind(TableFunction::BindInput &input) {
	const auto start = ConstantI64(input, 0);
	const auto stop = ConstantI64(input, 1);
	const auto step = ConstantI64(input, 2);
	if (step == int64_t(0)) {
		throw InvalidInputException("vs_series_step step must not be zero");
	}
	StepBind bind;
	if (start && stop && step) {
		bind = StepBind {*start, *step, StepCount(*start, *stop, *step)};
	}
	input.AddResultColumn("i", input.GetContext().GetFactory().CreateType(LogicalTypeId::BIGINT));
	input.SetCardinality(static_cast<idx_t>(bind.count), true);
	input.SetBindData<StepBind>(bind);
}

void SeriesStepInitLocal(TableFunction::InitLocalInput &input) {
	input.SetLocalState<StepScan>();
}

void SeriesStepExec(TableFunction::ExecInput &input) {
	const auto &bind = input.GetBindData<StepBind>();
	auto &scan = input.GetLocalState<StepScan>();
	auto chunk = input.GetOutputChunk();
	const auto rows = static_cast<idx_t>(std::min<uint64_t>(bind.count - scan.position, chunk.GetCapacity()));
	const auto first = scan.position;
	FillColumn<int64_t>(chunk, 0, rows, [&](idx_t row) {
		return static_cast<int64_t>(static_cast<uint64_t>(bind.start) +
		                            (first + row) * static_cast<uint64_t>(bind.step));
	});
	chunk.GetVector(0).SetSize(rows);
	scan.position += rows;
}

// The pushed-down filters narrow [lo, hi), possibly over several pushdown calls.
struct FilterBind {
	int64_t count = 0;
	int64_t lo = 0;
	int64_t hi = 0;
};

// The rows [position, end) left to produce.
struct FilterScan {
	int64_t position = 0;
	int64_t end = 0;
};

auto SaturatingIncrement(int64_t value) -> int64_t {
	return value == std::numeric_limits<int64_t>::max() ? value : value + 1;
}

void SeriesFilterBind(TableFunction::BindInput &input) {
	const auto count = BindSeries(input).count;
	input.SetBindData<FilterBind>(FilterBind {count, 0, count});
}

void SeriesFilterInitLocal(TableFunction::InitLocalInput &input) {
	const auto &bind = input.GetBindData<FilterBind>();
	input.SetLocalState<FilterScan>(FilterScan {bind.lo, std::min(bind.hi, bind.count)});
}

void SeriesFilterExec(TableFunction::ExecInput &input) {
	auto &scan = input.GetLocalState<FilterScan>();
	auto chunk = input.GetOutputChunk();
	const auto rows = ChunkRows(chunk, scan.end - scan.position);
	FillSeries(chunk, rows, scan.position);
	scan.position += static_cast<int64_t>(rows);
}

void SeriesFilterPushdown(TableFunction::FilterPushdownInput &input) {
	auto &bind = input.GetBindData<FilterBind>();
	const auto filter_count = input.GetFilterCount();
	for (idx_t index = 0; index < filter_count; index++) {
		auto filter = input.GetFilter(index);
		const auto comparison = filter.GetType();
		if (comparison != ExpressionType::COMPARE_LESSTHAN && comparison != ExpressionType::COMPARE_LESSTHANOREQUALTO &&
		    comparison != ExpressionType::COMPARE_GREATERTHAN &&
		    comparison != ExpressionType::COMPARE_GREATERTHANOREQUALTO && comparison != ExpressionType::COMPARE_EQUAL) {
			continue;
		}
		auto column = filter.GetChild(0);
		auto bound_expression = filter.GetChild(1);
		if (column.GetType() != ExpressionType::BOUND_COLUMN_REF ||
		    bound_expression.GetType() != ExpressionType::VALUE_CONSTANT ||
		    input.GetColumnIndex(column.GetColumnIndex()) != 0 ||
		    bound_expression.GetReturnType().GetTypeId() != LogicalTypeId::BIGINT) {
			continue;
		}
		auto bound_value = bound_expression.GetConstantValue();
		if (bound_value.IsNull()) {
			continue;
		}
		const auto bound = bound_value.Get<int64_t>();
		std::optional<int64_t> lo;
		std::optional<int64_t> hi;
		switch (comparison) {
		case ExpressionType::COMPARE_LESSTHAN:
			hi = bound;
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			hi = SaturatingIncrement(bound);
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
			lo = SaturatingIncrement(bound);
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			lo = bound;
			break;
		default:
			lo = bound;
			hi = SaturatingIncrement(bound);
			break;
		}
		if (lo) {
			bind.lo = std::max(bind.lo, *lo);
		}
		if (hi) {
			bind.hi = std::min(bind.hi, *hi);
		}
		input.Accept(index);
	}
}

// The next row a scanning thread claims.
struct ParScan {
	std::atomic<int64_t> next {0};
};

void SeriesParBind(TableFunction::BindInput &input) {
	input.SetBindData<CountBind>(BindSeries(input));
	input.SetOrderPreservation(OrderPreservation::NO_ORDER);
}

void SeriesParInitGlobal(TableFunction::InitGlobalInput &input) {
	const auto &bind = input.GetBindData<CountBind>();
	input.SetGlobalState<ParScan>();
	const auto blocks = (bind.count + PAR_BLOCK_ROWS - 1) / PAR_BLOCK_ROWS;
	input.SetMaxThreads(static_cast<idx_t>(std::max<int64_t>(blocks, 1)));
}

void SeriesParExec(TableFunction::ExecInput &input) {
	const auto &bind = input.GetBindData<CountBind>();
	auto &scan = input.GetGlobalState<ParScan>();
	auto chunk = input.GetOutputChunk();
	const auto first = scan.next.fetch_add(PAR_BLOCK_ROWS, std::memory_order_relaxed);
	const auto rows = ChunkRows(chunk, std::min(bind.count - first, PAR_BLOCK_ROWS));
	FillSeries(chunk, rows, first);
}

auto CreateCountFunction(Connection &conn, const std::string &name) -> TableFunction {
	auto function = TableFunction::Create(conn.GetFactory());
	function.SetName(name);
	function.GetSignature().AddParameter("n", conn.GetFactory().CreateType(LogicalTypeId::BIGINT));
	return function;
}

} // namespace

VERSUS_FAMILY(table) {
	auto &factory = conn.GetFactory();
	const auto bigint = factory.CreateType(LogicalTypeId::BIGINT);

	auto series = CreateCountFunction(conn, "vs_series");
	series.SetBindCallback(SeriesBind).SetInitLocalCallback(InitSeriesScan).SetExecCallback(SeriesExec);
	conn.Register(series);

	auto grid = CreateCountFunction(conn, "vs_grid");
	grid.SetBindCallback(GridBind).SetInitLocalCallback(InitSeriesScan).SetExecCallback(GridExec);
	grid.SetProjectionPushdown(true);
	conn.Register(grid);

	auto words = TableFunction::Create(factory);
	words.SetName("vs_words");
	words.GetSignature().AddParameter("n", bigint).AddParameter("len", factory.CreateType(LogicalTypeId::INTEGER));
	words.SetBindCallback(WordsBind_).SetInitLocalCallback(WordsInitLocal).SetExecCallback(WordsExec);
	conn.Register(words);

	auto repeat_row = TableFunction::Create(factory);
	repeat_row.SetName("vs_repeat_row");
	repeat_row.GetSignature()
	    .AddParameter("s", factory.CreateType(LogicalTypeId::VARCHAR))
	    .AddParameter("k", bigint)
	    .AddParameter("n", bigint);
	repeat_row.SetBindCallback(RepeatRowBind).SetInitLocalCallback(InitSeriesScan).SetExecCallback(RepeatRowExec);
	repeat_row.SetProjectionPushdown(true);
	conn.Register(repeat_row);

	auto series_step = TableFunction::Create(factory);
	series_step.SetName("vs_series_step");
	series_step.GetSignature()
	    .AddParameter("start", bigint)
	    .AddParameter("stop", bigint)
	    .AddParameter("step", bigint, Value::Create(factory, int64_t {1}), FunctionParameterKind::NAMED_ONLY);
	series_step.SetBindCallback(SeriesStepBind)
	    .SetInitLocalCallback(SeriesStepInitLocal)
	    .SetExecCallback(SeriesStepExec);
	conn.Register(series_step);

	auto series_filter = CreateCountFunction(conn, "vs_series_filter");
	series_filter.SetBindCallback(SeriesFilterBind)
	    .SetInitLocalCallback(SeriesFilterInitLocal)
	    .SetExecCallback(SeriesFilterExec)
	    .SetFilterPushdownCallback(SeriesFilterPushdown);
	conn.Register(series_filter);

	auto series_par = CreateCountFunction(conn, "vs_series_par");
	series_par.SetBindCallback(SeriesParBind).SetInitGlobalCallback(SeriesParInitGlobal).SetExecCallback(SeriesParExec);
	conn.Register(series_par);
}
