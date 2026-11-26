// Numeric scalar functions; the Rust twin is `../rust/scalar_num.rs`. NULL in any argument gives NULL unless noted.
//
// - vs_add_one(BIGINT) -> BIGINT: wrapping v + 1.
// - vs_clamp(v DOUBLE, lo DOUBLE, hi DOUBLE) -> DOUBLE: min(max(v, lo), hi).
// - vs_is_even(BIGINT) -> BOOLEAN: v % 2 = 0.
// - vs_coalesce_zero(BIGINT) -> BIGINT: NULL becomes 0, through special NULL handling.
// - vs_checked_div(a BIGINT, b BIGINT) -> BIGINT: a / b truncated, NULL for b = 0 or overflow.
// - vs_mix(a INTEGER, b BIGINT, c DOUBLE, d SMALLINT) -> DOUBLE: a * c + b - d in double.
// - vs_strict_sqrt(DOUBLE) -> DOUBLE: sqrt(v), an error for negative v.
// - vs_scale(v BIGINT, factor BIGINT) -> BIGINT: wrapping v * factor; factor is a constant read at bind.
#include "registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

using namespace duckdb::cxx;

namespace {

void AddOneExec(ScalarFunction::ExecInput &input) {
	ScalarExecutor::Execute<int64_t, int64_t>(
	    input, [](int64_t value) { return static_cast<int64_t>(static_cast<uint64_t>(value) + 1); });
}

void RegisterAddOne(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto bigint = factory.ParseType("BIGINT");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_add_one").SetExecCallback(AddOneExec);
	function.GetSignature().AddParameter("input", bigint).SetReturnType(bigint);
	conn.Register(function);
}

void ClampExec(ScalarFunction::ExecInput &input) {
	ScalarExecutor::Execute<double, double, double, double>(
	    input, [](double v, double lo, double hi) { return std::min(std::max(v, lo), hi); });
}

void RegisterClamp(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto dbl = factory.ParseType("DOUBLE");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_clamp").SetExecCallback(ClampExec);
	function.GetSignature().AddParameter("v", dbl).AddParameter("lo", dbl).AddParameter("hi", dbl).SetReturnType(dbl);
	conn.Register(function);
}

void IsEvenExec(ScalarFunction::ExecInput &input) {
	ScalarExecutor::Execute<bool, int64_t>(input, [](int64_t value) { return value % 2 == 0; });
}

void RegisterIsEven(Connection &conn) {
	auto &factory = conn.GetFactory();
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_is_even").SetExecCallback(IsEvenExec);
	function.GetSignature()
	    .AddParameter("input", factory.ParseType("BIGINT"))
	    .SetReturnType(factory.ParseType("BOOLEAN"));
	conn.Register(function);
}

void CoalesceZeroExec(ScalarFunction::ExecInput &input) {
	ScalarExecutor::Execute<int64_t, std::optional<int64_t>>(
	    input, [](std::optional<int64_t> value) { return value.value_or(0); });
}

void RegisterCoalesceZero(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto bigint = factory.ParseType("BIGINT");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_coalesce_zero")
	    .SetExecCallback(CoalesceZeroExec)
	    .SetNullHandling(FunctionNullHandling::SPECIAL);
	function.GetSignature().AddParameter("input", bigint).SetReturnType(bigint);
	conn.Register(function);
}

void CheckedDivExec(ScalarFunction::ExecInput &input) {
	ScalarExecutor::Execute<std::optional<int64_t>, int64_t, int64_t>(
	    input, [](int64_t a, int64_t b) -> std::optional<int64_t> {
		    if (b == 0 || (a == std::numeric_limits<int64_t>::min() && b == -1)) {
			    return std::nullopt;
		    }
		    return a / b;
	    });
}

void RegisterCheckedDiv(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto bigint = factory.ParseType("BIGINT");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_checked_div").SetExecCallback(CheckedDivExec);
	function.GetSignature().AddParameter("a", bigint).AddParameter("b", bigint).SetReturnType(bigint);
	conn.Register(function);
}

void MixExec(ScalarFunction::ExecInput &input) {
	ScalarExecutor::Execute<double, int32_t, int64_t, double, int16_t>(
	    input, [](int32_t a, int64_t b, double c, int16_t d) {
		    return static_cast<double>(a) * c + static_cast<double>(b) - static_cast<double>(d);
	    });
}

void RegisterMix(Connection &conn) {
	auto &factory = conn.GetFactory();
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_mix").SetExecCallback(MixExec);
	function.GetSignature()
	    .AddParameter("a", factory.ParseType("INTEGER"))
	    .AddParameter("b", factory.ParseType("BIGINT"))
	    .AddParameter("c", factory.ParseType("DOUBLE"))
	    .AddParameter("d", factory.ParseType("SMALLINT"))
	    .SetReturnType(factory.ParseType("DOUBLE"));
	conn.Register(function);
}

void StrictSqrtExec(ScalarFunction::ExecInput &input) {
	ScalarExecutor::Execute<double, double>(input, [](double value) {
		if (value < 0.0) {
			throw InvalidInputException("vs_strict_sqrt: negative input");
		}
		return std::sqrt(value);
	});
}

void RegisterStrictSqrt(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto dbl = factory.ParseType("DOUBLE");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_strict_sqrt").SetExecCallback(StrictSqrtExec);
	function.GetSignature().AddParameter("input", dbl).SetReturnType(dbl);
	conn.Register(function);
}

// The constant factor; nullopt when it is NULL.
struct ScaleData {
	std::optional<int64_t> factor;
	bool operator==(const ScaleData &other) const {
		return factor == other.factor;
	}
};

void ScaleBind(ScalarFunction::BindInput &input) {
	const auto factor = input.TryGetConstantArgument(1);
	if (!factor) {
		throw InvalidInputException("vs_scale: factor must be a constant");
	}
	input.SetBindData<ScaleData>(
	    ScaleData {factor->IsNull() ? std::nullopt : std::optional<int64_t>(factor->Get<int64_t>())});
}

void ScaleExec(ScalarFunction::ExecInput &input) {
	const auto factor = input.GetBindData<ScaleData>().factor;
	auto result = input.GetResult();
	ScalarExecutor::Execute<std::optional<int64_t>, int64_t>(
	    input.GetArg(0), result, input.GetRowCount(), [factor](int64_t value) -> std::optional<int64_t> {
		    if (!factor) {
			    return std::nullopt;
		    }
		    return static_cast<int64_t>(static_cast<uint64_t>(value) * static_cast<uint64_t>(*factor));
	    });
}

void RegisterScale(Connection &conn) {
	auto &factory = conn.GetFactory();
	const auto bigint = factory.ParseType("BIGINT");
	auto function = ScalarFunction::Create(factory);
	function.SetName("vs_scale").SetBindCallback(ScaleBind).SetExecCallback(ScaleExec);
	function.GetSignature().AddParameter("v", bigint).AddParameter("factor", bigint).SetReturnType(bigint);
	conn.Register(function);
}

} // namespace

VERSUS_FAMILY(scalar_num) {
	RegisterAddOne(conn);
	RegisterClamp(conn);
	RegisterIsEven(conn);
	RegisterCoalesceZero(conn);
	RegisterCheckedDiv(conn);
	RegisterMix(conn);
	RegisterStrictSqrt(conn);
	RegisterScale(conn);
}
