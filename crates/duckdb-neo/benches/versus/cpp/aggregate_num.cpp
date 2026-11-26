// Aggregates with plain states:
// - vs_sum(v BIGINT) -> BIGINT: wrapping sum, NULL without non-NULL rows.
// - vs_avg(d DOUBLE) -> DOUBLE: mean of the non-NULL rows, NULL without any.
// - vs_count_if(b BOOLEAN) -> BIGINT: number of true rows, 0 without any.
// - vs_spread(v BIGINT) -> BIGINT: wrapping max - min of the non-NULL rows, NULL without any.
// - vs_wavg(v DOUBLE, w DOUBLE) -> DOUBLE: sum(v * w) / sum(w) over rows where both are non-NULL, NULL without any.
// - vs_sum_scaled(v BIGINT, factor BIGINT) -> BIGINT: wrapping sum of v * factor, NULL without non-NULL v; factor must
//   be a non-NULL constant and is read once at bind time.
#include "registry.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>

using namespace duckdb::cxx;

namespace {

VectorView ReadableView(const Vector &vector) {
	if (vector.GetVectorType() == VectorType::OTHER) {
		vector.Flatten();
	}
	return vector.GetView();
}

// The engine sizes the result only after finalize, so NULLs go through the mask, not the size-checked SetNull.
struct ResultNulls {
	Vector &result;
	ValidityMask mask {nullptr};

	void Set(idx_t row) {
		if (!mask.words) {
			mask = result.GetValidityMutable();
		}
		mask.SetInvalid(row);
	}
};

int64_t WrappingAdd(int64_t a, int64_t b) {
	return static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
}

int64_t WrappingSub(int64_t a, int64_t b) {
	return static_cast<int64_t>(static_cast<uint64_t>(a) - static_cast<uint64_t>(b));
}

int64_t WrappingMul(int64_t a, int64_t b) {
	return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}

template <class STATE>
void PlainStateSize(AggregateFunction::SizeInput &input) {
	input.SetStateSize(sizeof(STATE));
}

template <class STATE>
void PlainStateInit(AggregateFunction::InitInput &input) {
	auto states = input.GetStates();
	const auto count = input.GetStateCount();
	for (idx_t i = 0; i < count; i++) {
		new (states[i]) STATE();
	}
}

struct SumState {
	int64_t sum = 0;
	uint64_t rows = 0;
};

void SumUpdate(AggregateFunction::UpdateInput &input) {
	const auto arg = input.GetArg(0);
	const auto values = ReadableView(arg);
	auto states = input.GetStates();
	const auto count = input.GetRowCount();
	for (idx_t i = 0; i < count; i++) {
		const auto entry = values.SelAt(i);
		if (values.RowIsValid(entry)) {
			auto &state = *static_cast<SumState *>(states[i]);
			state.sum = WrappingAdd(state.sum, values.Data<int64_t>()[entry]);
			state.rows++;
		}
	}
}

void SumCombine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	const auto count = input.GetStateCount();
	for (idx_t i = 0; i < count; i++) {
		const auto &source = *static_cast<const SumState *>(sources[i]);
		auto &target = *static_cast<SumState *>(targets[i]);
		target.sum = WrappingAdd(target.sum, source.sum);
		target.rows += source.rows;
	}
}

void SumFinalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	auto *out = result.GetDataMutable<int64_t>();
	ResultNulls nulls {result};
	const auto offset = input.GetResultOffset();
	const auto count = input.GetStateCount();
	for (idx_t i = 0; i < count; i++) {
		const auto &state = *static_cast<const SumState *>(states[i]);
		if (state.rows > 0) {
			out[offset + i] = state.sum;
		} else {
			nulls.Set(offset + i);
		}
	}
}

void RegisterSum(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto bigint = factory.ParseType("BIGINT");
	auto function = AggregateFunction::Create(factory);
	function.SetName("vs_sum")
	    .SetSizeCallback(PlainStateSize<SumState>)
	    .SetInitCallback(PlainStateInit<SumState>)
	    .SetUpdateCallback(SumUpdate)
	    .SetCombineCallback(SumCombine)
	    .SetFinalizeCallback(SumFinalize);
	function.GetSignature().AddParameter("v", bigint).SetReturnType(bigint);
	conn.Register(function);
}

struct AvgState {
	double sum = 0;
	uint64_t rows = 0;
};

void AvgUpdate(AggregateFunction::UpdateInput &input) {
	const auto arg = input.GetArg(0);
	const auto values = ReadableView(arg);
	auto states = input.GetStates();
	const auto count = input.GetRowCount();
	for (idx_t i = 0; i < count; i++) {
		const auto entry = values.SelAt(i);
		if (values.RowIsValid(entry)) {
			auto &state = *static_cast<AvgState *>(states[i]);
			state.sum += values.Data<double>()[entry];
			state.rows++;
		}
	}
}

void AvgCombine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	const auto count = input.GetStateCount();
	for (idx_t i = 0; i < count; i++) {
		const auto &source = *static_cast<const AvgState *>(sources[i]);
		auto &target = *static_cast<AvgState *>(targets[i]);
		target.sum += source.sum;
		target.rows += source.rows;
	}
}

void AvgFinalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	auto *out = result.GetDataMutable<double>();
	ResultNulls nulls {result};
	const auto offset = input.GetResultOffset();
	const auto count = input.GetStateCount();
	for (idx_t i = 0; i < count; i++) {
		const auto &state = *static_cast<const AvgState *>(states[i]);
		if (state.rows > 0) {
			out[offset + i] = state.sum / static_cast<double>(state.rows);
		} else {
			nulls.Set(offset + i);
		}
	}
}

void RegisterAvg(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto dbl = factory.ParseType("DOUBLE");
	auto function = AggregateFunction::Create(factory);
	function.SetName("vs_avg")
	    .SetSizeCallback(PlainStateSize<AvgState>)
	    .SetInitCallback(PlainStateInit<AvgState>)
	    .SetUpdateCallback(AvgUpdate)
	    .SetCombineCallback(AvgCombine)
	    .SetFinalizeCallback(AvgFinalize);
	function.GetSignature().AddParameter("d", dbl).SetReturnType(dbl);
	conn.Register(function);
}

struct CountIfState {
	int64_t count = 0;
};

void CountIfUpdate(AggregateFunction::UpdateInput &input) {
	const auto arg = input.GetArg(0);
	const auto values = ReadableView(arg);
	auto states = input.GetStates();
	const auto count = input.GetRowCount();
	for (idx_t i = 0; i < count; i++) {
		const auto entry = values.SelAt(i);
		if (values.RowIsValid(entry) && values.Data<bool>()[entry]) {
			static_cast<CountIfState *>(states[i])->count++;
		}
	}
}

void CountIfCombine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	const auto count = input.GetStateCount();
	for (idx_t i = 0; i < count; i++) {
		static_cast<CountIfState *>(targets[i])->count += static_cast<const CountIfState *>(sources[i])->count;
	}
}

void CountIfFinalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	auto *out = result.GetDataMutable<int64_t>();
	const auto offset = input.GetResultOffset();
	const auto count = input.GetStateCount();
	for (idx_t i = 0; i < count; i++) {
		out[offset + i] = static_cast<const CountIfState *>(states[i])->count;
	}
}

void RegisterCountIf(Connection &conn) {
	auto &factory = conn.GetFactory();
	auto function = AggregateFunction::Create(factory);
	function.SetName("vs_count_if")
	    .SetSizeCallback(PlainStateSize<CountIfState>)
	    .SetInitCallback(PlainStateInit<CountIfState>)
	    .SetUpdateCallback(CountIfUpdate)
	    .SetCombineCallback(CountIfCombine)
	    .SetFinalizeCallback(CountIfFinalize);
	function.GetSignature().AddParameter("b", factory.ParseType("BOOLEAN")).SetReturnType(factory.ParseType("BIGINT"));
	conn.Register(function);
}

struct SpreadState {
	int64_t min = std::numeric_limits<int64_t>::max();
	int64_t max = std::numeric_limits<int64_t>::min();
	uint64_t rows = 0;
};

void SpreadUpdate(AggregateFunction::UpdateInput &input) {
	const auto arg = input.GetArg(0);
	const auto values = ReadableView(arg);
	auto states = input.GetStates();
	const auto count = input.GetRowCount();
	for (idx_t i = 0; i < count; i++) {
		const auto entry = values.SelAt(i);
		if (values.RowIsValid(entry)) {
			const auto value = values.Data<int64_t>()[entry];
			auto &state = *static_cast<SpreadState *>(states[i]);
			state.min = std::min(state.min, value);
			state.max = std::max(state.max, value);
			state.rows++;
		}
	}
}

void SpreadCombine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	const auto count = input.GetStateCount();
	for (idx_t i = 0; i < count; i++) {
		const auto &source = *static_cast<const SpreadState *>(sources[i]);
		auto &target = *static_cast<SpreadState *>(targets[i]);
		target.min = std::min(target.min, source.min);
		target.max = std::max(target.max, source.max);
		target.rows += source.rows;
	}
}

void SpreadFinalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	auto *out = result.GetDataMutable<int64_t>();
	ResultNulls nulls {result};
	const auto offset = input.GetResultOffset();
	const auto count = input.GetStateCount();
	for (idx_t i = 0; i < count; i++) {
		const auto &state = *static_cast<const SpreadState *>(states[i]);
		if (state.rows > 0) {
			out[offset + i] = WrappingSub(state.max, state.min);
		} else {
			nulls.Set(offset + i);
		}
	}
}

void RegisterSpread(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto bigint = factory.ParseType("BIGINT");
	auto function = AggregateFunction::Create(factory);
	function.SetName("vs_spread")
	    .SetSizeCallback(PlainStateSize<SpreadState>)
	    .SetInitCallback(PlainStateInit<SpreadState>)
	    .SetUpdateCallback(SpreadUpdate)
	    .SetCombineCallback(SpreadCombine)
	    .SetFinalizeCallback(SpreadFinalize);
	function.GetSignature().AddParameter("v", bigint).SetReturnType(bigint);
	conn.Register(function);
}

struct WavgState {
	double weighted_sum = 0;
	double weight_sum = 0;
	uint64_t rows = 0;
};

void WavgUpdate(AggregateFunction::UpdateInput &input) {
	const auto v_arg = input.GetArg(0);
	const auto w_arg = input.GetArg(1);
	const auto v = ReadableView(v_arg);
	const auto w = ReadableView(w_arg);
	auto states = input.GetStates();
	const auto count = input.GetRowCount();
	for (idx_t i = 0; i < count; i++) {
		const auto v_entry = v.SelAt(i);
		const auto w_entry = w.SelAt(i);
		if (v.RowIsValid(v_entry) && w.RowIsValid(w_entry)) {
			const auto value = v.Data<double>()[v_entry];
			const auto weight = w.Data<double>()[w_entry];
			auto &state = *static_cast<WavgState *>(states[i]);
			state.weighted_sum += value * weight;
			state.weight_sum += weight;
			state.rows++;
		}
	}
}

void WavgCombine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	const auto count = input.GetStateCount();
	for (idx_t i = 0; i < count; i++) {
		const auto &source = *static_cast<const WavgState *>(sources[i]);
		auto &target = *static_cast<WavgState *>(targets[i]);
		target.weighted_sum += source.weighted_sum;
		target.weight_sum += source.weight_sum;
		target.rows += source.rows;
	}
}

void WavgFinalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	auto *out = result.GetDataMutable<double>();
	ResultNulls nulls {result};
	const auto offset = input.GetResultOffset();
	const auto count = input.GetStateCount();
	for (idx_t i = 0; i < count; i++) {
		const auto &state = *static_cast<const WavgState *>(states[i]);
		if (state.rows > 0) {
			out[offset + i] = state.weighted_sum / state.weight_sum;
		} else {
			nulls.Set(offset + i);
		}
	}
}

void RegisterWavg(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto dbl = factory.ParseType("DOUBLE");
	auto function = AggregateFunction::Create(factory);
	function.SetName("vs_wavg")
	    .SetSizeCallback(PlainStateSize<WavgState>)
	    .SetInitCallback(PlainStateInit<WavgState>)
	    .SetUpdateCallback(WavgUpdate)
	    .SetCombineCallback(WavgCombine)
	    .SetFinalizeCallback(WavgFinalize);
	function.GetSignature().AddParameter("v", dbl).AddParameter("w", dbl).SetReturnType(dbl);
	conn.Register(function);
}

struct ScaleFactor {
	int64_t value;
	bool operator==(const ScaleFactor &other) const {
		return value == other.value;
	}
};

void SumScaledBind(AggregateFunction::BindInput &input) {
	const auto factor = input.TryGetConstantArgument(1);
	if (!factor || factor->IsNull()) {
		throw InvalidInputException("vs_sum_scaled: factor must be a non-NULL constant");
	}
	input.SetBindData<ScaleFactor>(ScaleFactor {factor->Get<int64_t>()});
}

void SumScaledUpdate(AggregateFunction::UpdateInput &input) {
	const auto factor = input.GetBindData<ScaleFactor>().value;
	const auto arg = input.GetArg(0);
	const auto values = ReadableView(arg);
	auto states = input.GetStates();
	const auto count = input.GetRowCount();
	for (idx_t i = 0; i < count; i++) {
		const auto entry = values.SelAt(i);
		if (values.RowIsValid(entry)) {
			auto &state = *static_cast<SumState *>(states[i]);
			state.sum = WrappingAdd(state.sum, WrappingMul(values.Data<int64_t>()[entry], factor));
			state.rows++;
		}
	}
}

void RegisterSumScaled(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto bigint = factory.ParseType("BIGINT");
	auto function = AggregateFunction::Create(factory);
	function.SetName("vs_sum_scaled")
	    .SetBindCallback(SumScaledBind)
	    .SetSizeCallback(PlainStateSize<SumState>)
	    .SetInitCallback(PlainStateInit<SumState>)
	    .SetUpdateCallback(SumScaledUpdate)
	    .SetCombineCallback(SumCombine)
	    .SetFinalizeCallback(SumFinalize);
	function.GetSignature().AddParameter("v", bigint).AddParameter("factor", bigint).SetReturnType(bigint);
	conn.Register(function);
}

} // namespace

VERSUS_FAMILY(aggregate_num) {
	RegisterSum(conn);
	RegisterAvg(conn);
	RegisterCountIf(conn);
	RegisterSpread(conn);
	RegisterWavg(conn);
	RegisterSumScaled(conn);
}
