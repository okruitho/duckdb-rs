// Families register their `vs_*` functions through `VERSUS_FAMILY(<family>) { ... }` in `<family>.cpp`.
// The runner calls them in family-name order, so static initialization order does not matter.
#pragma once

#include "duckdb_cpp.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace versus {

using RegisterFamily = void (*)(duckdb::cxx::Connection &conn);

inline auto Families() -> std::map<std::string, RegisterFamily> & {
	static std::map<std::string, RegisterFamily> families;
	return families;
}

struct FamilyRegistrar {
	FamilyRegistrar(const char *name, RegisterFamily register_family) {
		if (!Families().emplace(name, register_family).second) {
			std::fprintf(stderr, "versus (cpp): family %s registered twice\n", name);
			std::abort();
		}
	}
};

inline auto RegisterAll(duckdb::cxx::Connection &conn) -> void {
	for (auto &family : Families()) {
		family.second(conn);
	}
}

} // namespace versus

// Family files are sources of the runner executable rather than of a library, so the linker keeps every registrar;
// `used` also keeps it under LTO.
#define VERSUS_FAMILY(family)                                                                                          \
	static void VersusRegister_##family(duckdb::cxx::Connection &conn);                                                \
	[[gnu::used]] static const ::versus::FamilyRegistrar versus_registrar_##family(#family, VersusRegister_##family);  \
	static void VersusRegister_##family(duckdb::cxx::Connection &conn)
