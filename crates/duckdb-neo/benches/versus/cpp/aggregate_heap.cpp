// Aggregates whose states own heap memory, so state construction, destruction and combine are part of every call:
//
// - vs_longest(VARCHAR) -> VARCHAR: the longest string in bytes, ties broken by the smallest in byte order; NULL
//   without non-NULL rows.
// - vs_distinct_count(BIGINT) -> BIGINT: the number of distinct non-NULL values.
// - vs_median(DOUBLE) -> DOUBLE: the lower median of the non-NULL values; NULL without any.
// - vs_sorted_list(BIGINT) -> BIGINT[]: the non-NULL values sorted ascending; NULL without any.
// - vs_top3(BIGINT) -> BIGINT[]: the three largest values, descending; NULL without any.
// - vs_concat_sorted(VARCHAR) -> VARCHAR: the non-NULL strings sorted in byte order and joined with ','; NULL without
//   any.
#include "registry.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

using namespace duckdb::cxx;

namespace {

auto ArgumentView(AggregateFunction::UpdateInput &input) -> VectorView {
	auto argument = input.GetArg(0);
	if (argument.GetVectorType() == VectorType::OTHER) {
		argument.Flatten();
	}
	return argument.GetView();
}

template <class STATE>
auto StateAt(void **states, idx_t index) -> STATE & {
	return *static_cast<STATE *>(states[index]);
}

template <class STATE>
void StateSize(AggregateFunction::SizeInput &input) {
	input.SetStateSize(sizeof(STATE));
}

template <class STATE>
void StateInit(AggregateFunction::InitInput &input) {
	auto states = input.GetStates();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		new (states[i]) STATE();
	}
}

template <class STATE>
void StateDestroy(AggregateFunction::DestroyInput &input) {
	auto states = input.GetStates();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		StateAt<STATE>(states, i).~STATE();
	}
}

template <class STATE>
void RegisterAggregate(Connection &conn, const char *name, const char *parameter_type, const char *return_type,
                       AggregateFunction::UpdateCallback update, AggregateFunction::CombineCallback combine,
                       AggregateFunction::FinalizeCallback finalize) {
	auto &factory = conn.GetFactory();
	auto function = AggregateFunction::Create(factory);
	function.SetName(name)
	    .SetSizeCallback(StateSize<STATE>)
	    .SetInitCallback(StateInit<STATE>)
	    .SetUpdateCallback(update)
	    .SetCombineCallback(combine)
	    .SetFinalizeCallback(finalize)
	    .SetOrderDependence(AggregateFunction::OrderDependence::INDEPENDENT);
	if constexpr (!std::is_trivially_destructible_v<STATE>) {
		function.SetDestroyCallback(StateDestroy<STATE>);
	}
	function.GetSignature()
	    .AddParameter("value", factory.ParseType(parameter_type))
	    .SetReturnType(factory.ParseType(return_type));
	conn.Register(function);
}

// Fetches the result's validity mask on the first NULL, so an all-valid result allocates none.
struct ResultNulls {
	Vector &result;
	ValidityMask mask {nullptr};

	void SetNull(idx_t row) {
		if (!mask.words) {
			mask = result.GetValidityMutable();
		}
		mask.SetInvalid(row);
	}
};

// The states hold bytes read from VARCHAR input, which is valid UTF-8, so the result skips validation.
void WriteString(Arena &heap, varchar_t *out, idx_t row, std::string_view value) {
	out[row] = heap.AddStringUnsafe(value);
}

struct LongestState {
	std::optional<std::string> value;
};

// Whether `candidate` beats `current`: longer in bytes, or as long and smaller in byte order.
auto IsLonger(std::string_view candidate, std::string_view current) -> bool {
	return candidate.size() > current.size() || (candidate.size() == current.size() && candidate < current);
}

void Offer(LongestState &state, std::string_view candidate) {
	if (state.value) {
		if (IsLonger(candidate, *state.value)) {
			state.value->assign(candidate);
		}
	} else {
		state.value.emplace(candidate);
	}
}

void LongestUpdate(AggregateFunction::UpdateInput &input) {
	const auto view = ArgumentView(input);
	const auto *values = view.Data<varchar_t>();
	auto states = input.GetStates();
	for (idx_t i = 0; i < input.GetRowCount(); i++) {
		if (view.IsValid(i)) {
			Offer(StateAt<LongestState>(states, i), values[view.SelAt(i)].view());
		}
	}
}

void LongestCombine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		const auto &source = StateAt<LongestState>(sources, i);
		if (source.value) {
			Offer(StateAt<LongestState>(targets, i), *source.value);
		}
	}
}

void LongestFinalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	ResultNulls nulls {result};
	auto heap = result.GetHeap();
	auto *out = result.GetDataMutable<varchar_t>();
	const auto offset = input.GetResultOffset();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		const auto &state = StateAt<LongestState>(states, i);
		if (state.value) {
			WriteString(heap, out, offset + i, *state.value);
		} else {
			nulls.SetNull(offset + i);
		}
	}
}

struct DistinctCountState {
	std::unordered_set<int64_t> values;
};

void DistinctCountUpdate(AggregateFunction::UpdateInput &input) {
	const auto view = ArgumentView(input);
	const auto *values = view.Data<int64_t>();
	auto states = input.GetStates();
	for (idx_t i = 0; i < input.GetRowCount(); i++) {
		if (view.IsValid(i)) {
			StateAt<DistinctCountState>(states, i).values.insert(values[view.SelAt(i)]);
		}
	}
}

void DistinctCountCombine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		const auto &source = StateAt<DistinctCountState>(sources, i);
		StateAt<DistinctCountState>(targets, i).values.insert(source.values.begin(), source.values.end());
	}
}

void DistinctCountFinalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	auto *out = result.GetDataMutable<int64_t>();
	const auto offset = input.GetResultOffset();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		out[offset + i] = static_cast<int64_t>(StateAt<DistinctCountState>(states, i).values.size());
	}
}

struct MedianState {
	std::vector<double> values;
};

void MedianUpdate(AggregateFunction::UpdateInput &input) {
	const auto view = ArgumentView(input);
	const auto *values = view.Data<double>();
	auto states = input.GetStates();
	for (idx_t i = 0; i < input.GetRowCount(); i++) {
		if (view.IsValid(i)) {
			StateAt<MedianState>(states, i).values.push_back(values[view.SelAt(i)]);
		}
	}
}

void MedianCombine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		const auto &source = StateAt<MedianState>(sources, i).values;
		auto &target = StateAt<MedianState>(targets, i).values;
		target.insert(target.end(), source.begin(), source.end());
	}
}

// Selecting in place only reorders the state's values, so a state finalized again gives the same result.
void MedianFinalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	ResultNulls nulls {result};
	auto *out = result.GetDataMutable<double>();
	const auto offset = input.GetResultOffset();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		auto &values = StateAt<MedianState>(states, i).values;
		if (values.empty()) {
			nulls.SetNull(offset + i);
			continue;
		}
		const auto middle = values.begin() + static_cast<std::ptrdiff_t>((values.size() - 1) / 2);
		std::nth_element(values.begin(), middle, values.end());
		out[offset + i] = *middle;
	}
}

struct SortedListState {
	std::vector<int64_t> values;
};

void SortedListUpdate(AggregateFunction::UpdateInput &input) {
	const auto view = ArgumentView(input);
	const auto *values = view.Data<int64_t>();
	auto states = input.GetStates();
	for (idx_t i = 0; i < input.GetRowCount(); i++) {
		if (view.IsValid(i)) {
			StateAt<SortedListState>(states, i).values.push_back(values[view.SelAt(i)]);
		}
	}
}

void SortedListCombine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		const auto &source = StateAt<SortedListState>(sources, i).values;
		auto &target = StateAt<SortedListState>(targets, i).values;
		target.insert(target.end(), source.begin(), source.end());
	}
}

// Window operators finalize into a result whose child already holds earlier rows, so append after them.
void SortedListFinalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	ResultNulls nulls {result};
	const auto offset = input.GetResultOffset();
	auto child = result.GetChild(0);
	auto child_offset = child.GetSize();
	idx_t elements = 0;
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		elements += StateAt<SortedListState>(states, i).values.size();
	}
	child.SetSize(child_offset + elements);
	auto *child_data = child.GetDataMutable<int64_t>();
	auto *entries = result.GetDataMutable<list_entry_t>();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		auto &values = StateAt<SortedListState>(states, i).values;
		if (values.empty()) {
			nulls.SetNull(offset + i);
			continue;
		}
		std::sort(values.begin(), values.end());
		std::copy(values.begin(), values.end(), child_data + child_offset);
		entries[offset + i] = list_entry_t {child_offset, values.size()};
		child_offset += values.size();
	}
}

// The largest values seen, descending, in `values[0, len)`.
struct Top3State {
	std::array<int64_t, 3> values {};
	idx_t len = 0;
};

void Offer(Top3State &state, int64_t value) {
	auto slot = state.len;
	if (state.len == state.values.size()) {
		if (value <= state.values[slot - 1]) {
			return;
		}
		slot--;
	} else {
		state.len++;
	}
	while (slot > 0 && state.values[slot - 1] < value) {
		state.values[slot] = state.values[slot - 1];
		slot--;
	}
	state.values[slot] = value;
}

void Top3Update(AggregateFunction::UpdateInput &input) {
	const auto view = ArgumentView(input);
	const auto *values = view.Data<int64_t>();
	auto states = input.GetStates();
	for (idx_t i = 0; i < input.GetRowCount(); i++) {
		if (view.IsValid(i)) {
			Offer(StateAt<Top3State>(states, i), values[view.SelAt(i)]);
		}
	}
}

void Top3Combine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		const auto &source = StateAt<Top3State>(sources, i);
		auto &target = StateAt<Top3State>(targets, i);
		for (idx_t j = 0; j < source.len; j++) {
			Offer(target, source.values[j]);
		}
	}
}

void Top3Finalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	ResultNulls nulls {result};
	const auto offset = input.GetResultOffset();
	auto child = result.GetChild(0);
	auto child_offset = child.GetSize();
	idx_t elements = 0;
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		elements += StateAt<Top3State>(states, i).len;
	}
	child.SetSize(child_offset + elements);
	auto *child_data = child.GetDataMutable<int64_t>();
	auto *entries = result.GetDataMutable<list_entry_t>();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		const auto &state = StateAt<Top3State>(states, i);
		if (state.len == 0) {
			nulls.SetNull(offset + i);
			continue;
		}
		std::copy(state.values.begin(), state.values.begin() + static_cast<std::ptrdiff_t>(state.len),
		          child_data + child_offset);
		entries[offset + i] = list_entry_t {child_offset, state.len};
		child_offset += state.len;
	}
}

struct ConcatSortedState {
	std::vector<std::string> values;
};

void ConcatSortedUpdate(AggregateFunction::UpdateInput &input) {
	const auto view = ArgumentView(input);
	const auto *values = view.Data<varchar_t>();
	auto states = input.GetStates();
	for (idx_t i = 0; i < input.GetRowCount(); i++) {
		if (view.IsValid(i)) {
			StateAt<ConcatSortedState>(states, i).values.emplace_back(values[view.SelAt(i)].view());
		}
	}
}

void ConcatSortedCombine(AggregateFunction::CombineInput &input) {
	auto sources = input.GetSources();
	auto targets = input.GetTargets();
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		const auto &source = StateAt<ConcatSortedState>(sources, i).values;
		auto &target = StateAt<ConcatSortedState>(targets, i).values;
		target.insert(target.end(), source.begin(), source.end());
	}
}

void ConcatSortedFinalize(AggregateFunction::FinalizeInput &input) {
	auto states = input.GetStates();
	auto result = input.GetResult();
	ResultNulls nulls {result};
	auto heap = result.GetHeap();
	auto *out = result.GetDataMutable<varchar_t>();
	const auto offset = input.GetResultOffset();
	std::string joined;
	for (idx_t i = 0; i < input.GetStateCount(); i++) {
		auto &values = StateAt<ConcatSortedState>(states, i).values;
		if (values.empty()) {
			nulls.SetNull(offset + i);
			continue;
		}
		std::sort(values.begin(), values.end());
		joined.clear();
		for (idx_t position = 0; position < values.size(); position++) {
			if (position > 0) {
				joined.push_back(',');
			}
			joined.append(values[position]);
		}
		WriteString(heap, out, offset + i, joined);
	}
}

} // namespace

VERSUS_FAMILY(aggregate_heap) {
	RegisterAggregate<LongestState>(conn, "vs_longest", "VARCHAR", "VARCHAR", LongestUpdate, LongestCombine,
	                                LongestFinalize);
	RegisterAggregate<DistinctCountState>(conn, "vs_distinct_count", "BIGINT", "BIGINT", DistinctCountUpdate,
	                                      DistinctCountCombine, DistinctCountFinalize);
	RegisterAggregate<MedianState>(conn, "vs_median", "DOUBLE", "DOUBLE", MedianUpdate, MedianCombine, MedianFinalize);
	RegisterAggregate<SortedListState>(conn, "vs_sorted_list", "BIGINT", "BIGINT[]", SortedListUpdate,
	                                   SortedListCombine, SortedListFinalize);
	RegisterAggregate<Top3State>(conn, "vs_top3", "BIGINT", "BIGINT[]", Top3Update, Top3Combine, Top3Finalize);
	RegisterAggregate<ConcatSortedState>(conn, "vs_concat_sorted", "VARCHAR", "VARCHAR", ConcatSortedUpdate,
	                                     ConcatSortedCombine, ConcatSortedFinalize);
}
