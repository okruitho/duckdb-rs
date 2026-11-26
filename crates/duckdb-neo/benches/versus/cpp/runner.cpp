// C++ runner of the versus benchmarks: runs one case against the functions in this directory.
// The arguments, the timing and the JSON line it prints are shared with `../../versus.rs`; see `../README.md`.

#include "duckdb_cpp.hpp"
#include "registry.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace duckdb::cxx;

namespace {

enum class Mode { CHECK, BENCH, LOOP };

struct Options {
	Mode mode = Mode::BENCH;
	std::string threads = "1";
	std::vector<std::string> setup;
	std::string check;
	std::string query;
	std::optional<std::string> reset;
	bool unprepared = false;
	uint64_t warmup_ms = 200;
	uint64_t samples = 30;
	uint64_t min_sample_ms = 2;
	double max_seconds = 3;
};

template <class T>
T ParseNumber(const std::string &flag, const std::string &value) {
	try {
		size_t end = 0;
		T result;
		if constexpr (std::is_floating_point_v<T>) {
			result = std::stod(value, &end);
		} else {
			if (value.empty() || value[0] == '-') {
				throw std::invalid_argument(value);
			}
			result = std::stoull(value, &end);
		}
		if (end != value.size()) {
			throw std::invalid_argument(value);
		}
		return result;
	} catch (const std::logic_error &) {
		throw std::runtime_error(flag + " expects a number, got \"" + value + "\"");
	}
}

Options ParseOptions(int argc, char **argv) {
	Options options;
	bool has_mode = false;
	for (int i = 1; i < argc; i++) {
		const std::string flag = argv[i];
		if (flag == "--unprepared") {
			options.unprepared = true;
			continue;
		}
		if (i + 1 >= argc) {
			throw std::runtime_error(flag + " expects a value");
		}
		const std::string value = argv[++i];
		if (flag == "--mode") {
			has_mode = true;
			if (value == "check") {
				options.mode = Mode::CHECK;
			} else if (value == "bench") {
				options.mode = Mode::BENCH;
			} else if (value == "loop") {
				options.mode = Mode::LOOP;
			} else {
				throw std::runtime_error("unknown mode \"" + value + "\"");
			}
		} else if (flag == "--threads") {
			options.threads = value;
		} else if (flag == "--setup") {
			options.setup.push_back(value);
		} else if (flag == "--check") {
			options.check = value;
		} else if (flag == "--query") {
			options.query = value;
		} else if (flag == "--reset") {
			options.reset = value;
		} else if (flag == "--warmup-ms") {
			options.warmup_ms = ParseNumber<uint64_t>(flag, value);
		} else if (flag == "--samples") {
			options.samples = ParseNumber<uint64_t>(flag, value);
		} else if (flag == "--min-sample-ms") {
			options.min_sample_ms = ParseNumber<uint64_t>(flag, value);
		} else if (flag == "--max-seconds") {
			options.max_seconds = ParseNumber<double>(flag, value);
		} else {
			throw std::runtime_error("unknown argument \"" + flag + "\"");
		}
	}
	if (!has_mode) {
		throw std::runtime_error("--mode is required");
	}
	if (options.check.empty() || options.query.empty()) {
		throw std::runtime_error("--check and --query are required");
	}
	return options;
}

std::string ReadCheck(Connection &conn, const std::string &sql) {
	auto result = conn.Execute(sql);
	auto chunk = result.FetchChunk();
	if (!chunk) {
		throw std::runtime_error("the check query returned no rows");
	}
	auto vector = chunk.GetVector(0);
	if (vector.GetLogicalType().GetTypeId() != LogicalTypeId::VARCHAR) {
		throw std::runtime_error("the check query must return VARCHAR, got " + vector.GetLogicalType().ToText());
	}
	const auto view = vector.GetView();
	if (view.count == 0) {
		throw std::runtime_error("the check query returned no rows");
	}
	if (!view.IsValid(0)) {
		return "NULL";
	}
	return std::string(view.Data<varchar_t>()[view.SelAt(0)].view());
}

struct Iteration {
	uint64_t ns;
	idx_t rows;
};

Iteration Iterate(Connection &conn, PreparedStatement *prepared, const std::string &query,
                  const std::optional<std::string> &reset) {
	if (reset) {
		conn.Execute(*reset).Drain();
	}
	const auto start = std::chrono::steady_clock::now();
	idx_t rows = 0;
	{
		auto result = prepared ? prepared->Execute() : conn.Execute(query);
		while (auto chunk = result.FetchChunk()) {
			rows += chunk.GetRowCount();
		}
	}
	const auto end = std::chrono::steady_clock::now();
	return {static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()), rows};
}

uint64_t ElapsedNs(std::chrono::steady_clock::time_point start) {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

uint64_t Median(std::vector<uint64_t> values) {
	std::sort(values.begin(), values.end());
	return values[values.size() / 2];
}

std::string JsonString(const std::string &value) {
	std::string out = "\"";
	for (const char c : value) {
		switch (c) {
		case '"':
			out += "\\\"";
			break;
		case '\\':
			out += "\\\\";
			break;
		case '\n':
			out += "\\n";
			break;
		case '\r':
			out += "\\r";
			break;
		case '\t':
			out += "\\t";
			break;
		default:
			if (static_cast<unsigned char>(c) < 0x20) {
				char escaped[8];
				std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(c));
				out += escaped;
			} else {
				out += c;
			}
		}
	}
	return out + "\"";
}

std::string Run(const Options &options) {
	Environment environment;
	Instance instance(environment);
	instance.GetConfig().SetOption("threads", options.threads, SettingScope::GLOBAL);
	instance.Attach(":memory:", true);
	auto conn = instance.Connect();
	versus::RegisterAll(conn);

	for (auto &sql : options.setup) {
		conn.Execute(sql).Drain();
	}
	const auto check = ReadCheck(conn, options.check);
	if (options.mode == Mode::CHECK) {
		return "{\"check\": " + JsonString(check) + "}";
	}

	std::optional<PreparedStatement> prepared;
	if (!options.unprepared) {
		auto statements = conn.ParseSQL(options.query);
		auto statement = statements.Next();
		if (!statement || statements.Next()) {
			throw std::runtime_error("--query must hold exactly one statement");
		}
		prepared.emplace(conn.Prepare(statement, false));
	}
	auto *statement = prepared ? &*prepared : nullptr;
	const auto max_ns = static_cast<uint64_t>(options.max_seconds * 1e9);

	uint64_t iters = 0;
	idx_t last_rows = 0;
	std::vector<std::string> samples_ns;
	if (options.mode == Mode::LOOP) {
		const auto start = std::chrono::steady_clock::now();
		while (iters == 0 || ElapsedNs(start) < max_ns) {
			last_rows = Iterate(conn, statement, options.query, options.reset).rows;
			iters++;
		}
	} else {
		const auto warmup_ns = options.warmup_ms * 1000000;
		std::vector<uint64_t> warmup;
		auto start = std::chrono::steady_clock::now();
		while (warmup.size() < 3 || ElapsedNs(start) < warmup_ns) {
			warmup.push_back(Iterate(conn, statement, options.query, options.reset).ns);
		}
		const auto typical_ns = std::max<uint64_t>(Median(warmup), 1);
		const auto min_sample_ns = options.min_sample_ms * 1000000;
		iters = options.reset ? 1 : std::clamp<uint64_t>((min_sample_ns + typical_ns - 1) / typical_ns, 1, 1000000);

		start = std::chrono::steady_clock::now();
		while (samples_ns.size() < options.samples) {
			uint64_t sample_ns = 0;
			for (uint64_t i = 0; i < iters; i++) {
				const auto iteration = Iterate(conn, statement, options.query, options.reset);
				sample_ns += iteration.ns;
				last_rows = iteration.rows;
			}
			char value[64];
			std::snprintf(value, sizeof(value), "%.1f", static_cast<double>(sample_ns) / static_cast<double>(iters));
			samples_ns.emplace_back(value);
			if (samples_ns.size() >= 5 && ElapsedNs(start) >= max_ns) {
				break;
			}
		}
	}
	std::string samples;
	for (size_t i = 0; i < samples_ns.size(); i++) {
		samples += (i ? ", " : "") + samples_ns[i];
	}
	return "{\"check\": " + JsonString(check) + ", \"iters\": " + std::to_string(iters) +
	       ", \"result_rows\": " + std::to_string(last_rows) + ", \"samples_ns\": [" + samples + "]}";
}

} // namespace

int main(int argc, char **argv) {
	try {
		const auto line = Run(ParseOptions(argc, argv));
		std::printf("%s\n", line.c_str());
		return 0;
	} catch (const std::exception &error) {
		std::fprintf(stderr, "versus (cpp): %s\n", error.what());
		return 1;
	}
}
