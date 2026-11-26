#include "registry.hpp"

#include <cstdint>

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

} // namespace

VERSUS_FAMILY(scalar_num) {
	RegisterAddOne(conn);
}
