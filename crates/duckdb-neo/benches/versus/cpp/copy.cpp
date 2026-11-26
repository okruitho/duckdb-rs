// The `copy` family on the C++ API; the duckdb-neo side, which documents both formats, is `../rust/copy.rs`.
//
// `vs_lines`: text, one row per line, fields separated by the `sep` option (default tab), NULL as `\N`, VARCHAR
// escaping `\`, newline and the separator; BIGINT, DOUBLE, BOOLEAN and VARCHAR columns.
// `vs_fixed`: binary BIGINT/DOUBLE: a header, then per row and column a validity byte and 8 little-endian bytes.
// Values are copied in host byte order, which is little-endian on the machines this runs on.

#include "registry.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

using namespace duckdb::cxx;

namespace {

constexpr idx_t LINES_BUFFER_SIZE = 1 << 20;
constexpr std::string_view NULL_FIELD = "\\N";
constexpr std::string_view FIXED_MAGIC = "VSFX";
constexpr idx_t FIXED_VALUE_WIDTH = 9;

FileHandle OpenOutput(Context &context, const std::string &file_path) {
	return context.GetFileSystem().OpenFile(file_path, {FileFlags::WRITE, FileFlags::FILE_CREATE_NEW});
}

void WriteAll(FileHandle &file, const std::string &bytes) {
	const char *data = bytes.data();
	idx_t left = bytes.size();
	while (left > 0) {
		const auto written = file.Write(data, left);
		if (written == 0) {
			throw InvalidInputException("copy output file accepted no bytes");
		}
		data += written;
		left -= written;
	}
}

InvalidInputException InvalidField(const char *format, const char *type_name, std::string_view field) {
	return InvalidInputException(std::string(format) + ": invalid " + type_name + " field '" + std::string(field) +
	                             "'");
}

bool EqualsIgnoreCase(std::string_view left, std::string_view right) {
	return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(), [](char a, char b) {
		       return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
	       });
}

//----------------------------------------------------------------------------------------------------------------------
// vs_lines
//----------------------------------------------------------------------------------------------------------------------

enum class LinesColumn : uint8_t { BIGINT, DOUBLE, BOOLEAN, VARCHAR };

LinesColumn LinesColumnFromType(const LogicalType &type) {
	switch (type.GetTypeId()) {
	case LogicalTypeId::BIGINT:
		return LinesColumn::BIGINT;
	case LogicalTypeId::DOUBLE:
		return LinesColumn::DOUBLE;
	case LogicalTypeId::BOOLEAN:
		return LinesColumn::BOOLEAN;
	case LogicalTypeId::VARCHAR:
		return LinesColumn::VARCHAR;
	default:
		throw InvalidInputException("vs_lines does not support " + type.ToText());
	}
}

struct LinesBind {
	char sep = '\t';
	std::vector<LinesColumn> columns;
	std::vector<LogicalType> types;
};

bool IsAsciiPunctuation(unsigned char byte) {
	return (byte >= 33 && byte <= 47) || (byte >= 58 && byte <= 64) || (byte >= 91 && byte <= 96) ||
	       (byte >= 123 && byte <= 126);
}

bool IsLinesSep(unsigned char byte) {
	return (IsAsciiPunctuation(byte) && byte != '\\' && byte != '-' && byte != '+' && byte != '.') || byte == ' ' ||
	       byte == '\t';
}

char LinesSep(const Value &value) {
	if (value.GetLogicalType().GetTypeId() == LogicalTypeId::VARCHAR && !value.IsNull()) {
		const auto text = value.Get<varchar_t>().view();
		if (text.size() == 1 && IsLinesSep(static_cast<unsigned char>(text[0]))) {
			return text[0];
		}
	}
	throw InvalidInputException(
	    "vs_lines: sep must be one ASCII punctuation or whitespace character other than \\, -, + and .");
}

template <class INPUT>
LinesBind BindLines(INPUT &input) {
	LinesBind bind;
	for (idx_t index = 0; index < input.GetOptionCount(); index++) {
		const auto name = input.GetOptionName(index);
		if (!EqualsIgnoreCase(name, "sep")) {
			throw InvalidInputException("vs_lines: unknown option " + name);
		}
		bind.sep = LinesSep(input.GetOptionValue(index));
	}
	for (idx_t index = 0; index < input.GetColumnCount(); index++) {
		auto type = input.GetColumnType(index);
		bind.columns.push_back(LinesColumnFromType(type));
		bind.types.push_back(std::move(type));
	}
	return bind;
}

void PushI64(int64_t value, std::string &out) {
	char digits[20];
	idx_t start = sizeof(digits);
	auto rest = value < 0 ? 0 - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
	do {
		digits[--start] = static_cast<char>('0' + rest % 10);
		rest /= 10;
	} while (rest != 0);
	if (value < 0) {
		out += '-';
	}
	out.append(digits + start, sizeof(digits) - start);
}

void PushF64(double value, std::string &out) {
	if (std::isnan(value)) {
		out += "nan";
		return;
	}
	if (std::isinf(value)) {
		out += value < 0 ? "-inf" : "inf";
		return;
	}
	char text[32];
	const auto end = std::to_chars(text, text + sizeof(text), value, std::chars_format::scientific, 16).ptr;
	// to_chars spells the exponent e+05 or e-05; the format spells it e5 or e-5.
	const auto exponent = std::find(text, end, 'e');
	out.append(text, exponent + 1);
	if (exponent[1] == '-') {
		out += '-';
	}
	auto digit = exponent + 2;
	while (digit + 1 < end && *digit == '0') {
		digit++;
	}
	out.append(digit, end);
}

void PushEscaped(std::string_view text, char sep, std::string &out) {
	for (const char byte : text) {
		if (byte == '\\') {
			out += "\\\\";
		} else if (byte == '\n') {
			out += "\\n";
		} else if (byte == sep) {
			out += '\\';
			out += sep;
		} else {
			out += byte;
		}
	}
}

void EncodeLinesField(LinesColumn column, const VectorView &view, idx_t row, char sep, std::string &out) {
	const auto element = view.SelAt(row);
	if (!view.RowIsValid(element)) {
		out += NULL_FIELD;
		return;
	}
	switch (column) {
	case LinesColumn::BIGINT:
		PushI64(view.Data<int64_t>()[element], out);
		break;
	case LinesColumn::DOUBLE:
		PushF64(view.Data<double>()[element], out);
		break;
	case LinesColumn::BOOLEAN:
		out += view.Data<bool>()[element] ? "true" : "false";
		break;
	case LinesColumn::VARCHAR:
		PushEscaped(view.Data<varchar_t>()[element].view(), sep, out);
		break;
	}
}

void EncodeLinesChunk(const LinesBind &bind, const DataChunk &chunk, std::string &out) {
	std::vector<VectorView> views;
	for (idx_t index = 0; index < bind.columns.size(); index++) {
		views.push_back(chunk.GetVector(index).GetView());
	}

	const auto rows = chunk.GetRowCount();
	for (idx_t row = 0; row < rows; row++) {
		for (idx_t index = 0; index < views.size(); index++) {
			if (index > 0) {
				out += bind.sep;
			}
			EncodeLinesField(bind.columns[index], views[index], row, bind.sep, out);
		}
		out += '\n';
	}
}

void LinesToBind(CopyFunction::CopyToBindInput &input) {
	input.SetBindData<LinesBind>(BindLines(input));
}

void LinesToInit(CopyFunction::CopyToInitInput &input) {
	input.SetInitData<FileHandle>(OpenOutput(input.GetContext(), input.GetFilePath()));
}

void LinesToBatch(CopyFunction::CopyToBatchInput &input) {
	const auto &bind = input.GetBindData<LinesBind>();
	auto batch = input.TakeBatch();
	auto shared = batch.CreateSharedScanState();
	auto worker = batch.CreateWorkerScanState();
	DataChunk chunk(input.GetContext().GetFactory(), bind.types);
	std::string encoded;
	while (batch.Scan(shared, worker, chunk)) {
		EncodeLinesChunk(bind, chunk, encoded);
	}
	input.SetBatchData<std::string>(std::move(encoded));
}

void LinesToFlush(CopyFunction::CopyToFlushInput &input) {
	WriteAll(input.GetInitData<FileHandle>(), input.GetBatchData<std::string>());
}

void LinesToFinalize(CopyFunction::CopyToFinalizeInput &input) {
	input.GetInitData<FileHandle>().Close();
}

struct LinesReadBind {
	std::string file_path;
	LinesBind lines;
};

struct FieldRange {
	idx_t begin;
	idx_t end;
};

void SplitLine(const char *data, idx_t line_begin, idx_t line_end, char sep, idx_t columns,
               std::vector<FieldRange> &fields) {
	auto field_begin = line_begin;
	idx_t count = 1;
	auto pos = line_begin;
	while (pos < line_end) {
		const auto byte = data[pos];
		if (byte == '\\') {
			pos += 2;
			continue;
		}
		if (byte == sep) {
			fields.push_back({field_begin, pos});
			count++;
			field_begin = pos + 1;
		}
		pos++;
	}
	fields.push_back({field_begin, line_end});
	if (count != columns) {
		throw InvalidInputException("vs_lines: a row has " + std::to_string(count) + " fields, expected " +
		                            std::to_string(columns));
	}
}

// A `vs_lines` read: the unparsed bytes [begin, end) of `buffer`, and the field ranges of the current chunk.
struct LinesReadState {
	FileHandle file;
	std::vector<char> buffer;
	idx_t begin = 0;
	idx_t end = 0;
	bool eof = false;
	std::vector<FieldRange> fields;
	std::string unescaped;

	explicit LinesReadState(FileHandle file) : file(std::move(file)), buffer(LINES_BUFFER_SIZE) {
	}

	std::string_view Field(idx_t index) const {
		return std::string_view(buffer.data() + fields[index].begin, fields[index].end - fields[index].begin);
	}

	// Reads more of the file behind the unparsed bytes; false at the end of the file.
	bool Refill() {
		if (eof) {
			if (begin < end) {
				throw InvalidInputException("vs_lines: the file does not end with a newline");
			}
			return false;
		}
		std::memmove(buffer.data(), buffer.data() + begin, end - begin);
		end -= begin;
		begin = 0;
		if (end == buffer.size()) {
			buffer.resize(buffer.size() * 2);
		}
		const auto read = file.Read(buffer.data() + end, buffer.size() - end);
		eof = read == 0;
		end += read;
		return true;
	}

	// Splits up to `capacity` complete lines into fields; refills only before the first one.
	idx_t SplitRows(idx_t capacity, const LinesBind &bind) {
		fields.clear();
		idx_t rows = 0;
		while (rows < capacity) {
			const auto *newline = static_cast<const char *>(std::memchr(buffer.data() + begin, '\n', end - begin));
			if (newline) {
				const auto line_end = static_cast<idx_t>(newline - buffer.data());
				SplitLine(buffer.data(), begin, line_end, bind.sep, bind.columns.size(), fields);
				begin = line_end + 1;
				rows++;
			} else if (rows == 0) {
				if (!Refill()) {
					break;
				}
			} else {
				break;
			}
		}
		return rows;
	}
};

bool ParseI64(std::string_view field, int64_t &out) {
	const bool negative = !field.empty() && field[0] == '-';
	const auto digits = negative ? field.substr(1) : field;
	if (digits.empty() || digits.size() > 19) {
		return false;
	}
	uint64_t value = 0;
	for (const char digit : digits) {
		if (digit < '0' || digit > '9') {
			return false;
		}
		value = value * 10 + static_cast<uint64_t>(digit - '0');
	}
	const uint64_t limit = negative ? uint64_t(1) << 63 : static_cast<uint64_t>(INT64_MAX);
	if (value > limit) {
		return false;
	}
	out = static_cast<int64_t>(negative ? 0 - value : value);
	return true;
}

bool ParseF64(std::string_view field, double &out) {
	const auto end = field.data() + field.size();
	const auto result = std::from_chars(field.data(), end, out);
	return result.ec == std::errc() && result.ptr == end;
}

bool ParseBool(std::string_view field, bool &out) {
	if (field == "true") {
		out = true;
		return true;
	}
	if (field == "false") {
		out = false;
		return true;
	}
	return false;
}

void Unescape(std::string_view field, char sep, std::string &out) {
	for (idx_t pos = 0; pos < field.size(); pos++) {
		const auto byte = field[pos];
		if (byte != '\\') {
			out += byte;
			continue;
		}
		if (++pos == field.size()) {
			throw InvalidField("vs_lines", "VARCHAR", field);
		}
		const auto escaped = field[pos];
		if (escaped == '\\') {
			out += '\\';
		} else if (escaped == 'n') {
			out += '\n';
		} else if (escaped == sep) {
			out += sep;
		} else {
			throw InvalidField("vs_lines", "VARCHAR", field);
		}
	}
}

// Fetches the validity mask on the first NULL, so output without NULLs allocates none.
struct OutputValidity {
	Vector &vec;
	ValidityMask mask {nullptr};

	void SetNull(idx_t row) {
		if (!mask.words) {
			mask = vec.GetValidityMutable();
		}
		mask.SetInvalid(row);
	}
};

template <class T, class PARSE>
void DecodeLinesColumn(DataChunk &output, idx_t index, idx_t rows, const LinesReadState &state, idx_t columns,
                       const char *type_name, PARSE parse) {
	auto vec = output.GetVector(index);
	vec.SetSize(rows);
	auto *data = vec.GetDataMutable<T>();
	OutputValidity validity {vec};
	for (idx_t row = 0; row < rows; row++) {
		const auto field = state.Field(row * columns + index);
		if (field == NULL_FIELD) {
			validity.SetNull(row);
			continue;
		}
		T value;
		if (!parse(field, value)) {
			throw InvalidField("vs_lines", type_name, field);
		}
		data[row] = value;
	}
}

void DecodeLinesVarchar(DataChunk &output, idx_t index, idx_t rows, LinesReadState &state, idx_t columns, char sep) {
	auto vec = output.GetVector(index);
	vec.SetSize(rows);
	OutputValidity validity {vec};
	for (idx_t row = 0; row < rows; row++) {
		const auto field = state.Field(row * columns + index);
		if (field == NULL_FIELD) {
			validity.SetNull(row);
			continue;
		}
		state.unescaped.clear();
		Unescape(field, sep, state.unescaped);
		vec.AssignString(row, state.unescaped);
	}
}

void LinesFromBind(CopyFunction::CopyFromBindInput &input) {
	input.SetBindData<LinesReadBind>(LinesReadBind {input.GetFilePath(), BindLines(input)});
}

void LinesFromInitLocal(CopyFunction::CopyFromInitLocalInput &input) {
	const auto &bind = input.GetBindData<LinesReadBind>();
	auto file = input.GetContext().GetFileSystem().OpenFile(bind.file_path, {FileFlags::READ});
	input.SetLocalState<LinesReadState>(std::move(file));
}

void LinesFromExec(CopyFunction::CopyFromExecInput &input) {
	const auto &bind = input.GetBindData<LinesReadBind>().lines;
	auto &state = input.GetLocalState<LinesReadState>();
	auto output = input.GetOutputChunk();
	const auto rows = state.SplitRows(output.GetCapacity(), bind);
	if (rows == 0) {
		return;
	}
	const auto columns = bind.columns.size();
	for (idx_t index = 0; index < columns; index++) {
		switch (bind.columns[index]) {
		case LinesColumn::BIGINT:
			DecodeLinesColumn<int64_t>(output, index, rows, state, columns, "BIGINT", ParseI64);
			break;
		case LinesColumn::DOUBLE:
			DecodeLinesColumn<double>(output, index, rows, state, columns, "DOUBLE", ParseF64);
			break;
		case LinesColumn::BOOLEAN:
			DecodeLinesColumn<bool>(output, index, rows, state, columns, "BOOLEAN", ParseBool);
			break;
		case LinesColumn::VARCHAR:
			DecodeLinesVarchar(output, index, rows, state, columns, bind.sep);
			break;
		}
	}
}

//----------------------------------------------------------------------------------------------------------------------
// vs_fixed
//----------------------------------------------------------------------------------------------------------------------

enum class FixedColumn : uint8_t { BIGINT = 0, DOUBLE = 1 };

FixedColumn FixedColumnFromType(const LogicalType &type) {
	switch (type.GetTypeId()) {
	case LogicalTypeId::BIGINT:
		return FixedColumn::BIGINT;
	case LogicalTypeId::DOUBLE:
		return FixedColumn::DOUBLE;
	default:
		throw InvalidInputException("vs_fixed does not support " + type.ToText());
	}
}

struct FixedBind {
	std::vector<FixedColumn> columns;
	std::vector<LogicalType> types;
};

template <class INPUT>
FixedBind BindFixed(INPUT &input) {
	if (input.GetOptionCount() > 0) {
		throw InvalidInputException("vs_fixed takes no options");
	}
	FixedBind bind;
	for (idx_t index = 0; index < input.GetColumnCount(); index++) {
		auto type = input.GetColumnType(index);
		bind.columns.push_back(FixedColumnFromType(type));
		bind.types.push_back(std::move(type));
	}
	return bind;
}

std::string FixedHeader(const std::vector<FixedColumn> &columns) {
	std::string header(FIXED_MAGIC);
	const auto count = static_cast<uint32_t>(columns.size());
	header.append(reinterpret_cast<const char *>(&count), sizeof(count));
	for (const auto column : columns) {
		header += static_cast<char>(column);
	}
	return header;
}

template <class T>
void PushFixed(const VectorView &view, idx_t row, std::string &out) {
	const auto element = view.SelAt(row);
	if (!view.RowIsValid(element)) {
		out.append(FIXED_VALUE_WIDTH, '\0');
		return;
	}
	out += '\1';
	out.append(reinterpret_cast<const char *>(&view.Data<T>()[element]), sizeof(T));
}

void EncodeFixedChunk(const std::vector<FixedColumn> &columns, const DataChunk &chunk, std::string &out) {
	std::vector<VectorView> views;
	for (idx_t index = 0; index < columns.size(); index++) {
		views.push_back(chunk.GetVector(index).GetView());
	}

	const auto rows = chunk.GetRowCount();
	for (idx_t row = 0; row < rows; row++) {
		for (idx_t index = 0; index < views.size(); index++) {
			switch (columns[index]) {
			case FixedColumn::BIGINT:
				PushFixed<int64_t>(views[index], row, out);
				break;
			case FixedColumn::DOUBLE:
				PushFixed<double>(views[index], row, out);
				break;
			}
		}
	}
}

void FixedToBind(CopyFunction::CopyToBindInput &input) {
	input.SetBindData<FixedBind>(BindFixed(input));
}

void FixedToInit(CopyFunction::CopyToInitInput &input) {
	auto file = OpenOutput(input.GetContext(), input.GetFilePath());
	WriteAll(file, FixedHeader(input.GetBindData<FixedBind>().columns));
	input.SetInitData<FileHandle>(std::move(file));
}

void FixedToBatch(CopyFunction::CopyToBatchInput &input) {
	const auto &bind = input.GetBindData<FixedBind>();
	auto batch = input.TakeBatch();
	auto shared = batch.CreateSharedScanState();
	auto worker = batch.CreateWorkerScanState();
	DataChunk chunk(input.GetContext().GetFactory(), bind.types);
	std::string encoded;
	while (batch.Scan(shared, worker, chunk)) {
		EncodeFixedChunk(bind.columns, chunk, encoded);
	}
	input.SetBatchData<std::string>(std::move(encoded));
}

void FixedToFlush(CopyFunction::CopyToFlushInput &input) {
	WriteAll(input.GetInitData<FileHandle>(), input.GetBatchData<std::string>());
}

void FixedToFinalize(CopyFunction::CopyToFinalizeInput &input) {
	input.GetInitData<FileHandle>().Close();
}

struct FixedReadBind {
	std::string file_path;
	FixedBind fixed;
};

// A `vs_fixed` read: the offset and number of the rows not read yet, and the current chunk's bytes.
struct FixedReadState {
	FileHandle file;
	idx_t offset;
	idx_t rows_left;
	std::vector<char> buffer;

	FixedReadState(FileHandle file, idx_t offset, idx_t rows_left)
	    : file(std::move(file)), offset(offset), rows_left(rows_left) {
	}
};

template <class T>
void DecodeFixedColumn(DataChunk &output, idx_t index, idx_t rows, const std::vector<char> &data) {
	const auto row_width = data.size() / rows;
	auto vec = output.GetVector(index);
	vec.SetSize(rows);
	auto *values = vec.GetDataMutable<T>();
	OutputValidity validity {vec};
	for (idx_t row = 0; row < rows; row++) {
		const auto *field = data.data() + row * row_width + index * FIXED_VALUE_WIDTH;
		switch (field[0]) {
		case 0:
			validity.SetNull(row);
			break;
		case 1:
			std::memcpy(&values[row], field + 1, sizeof(T));
			break;
		default:
			throw InvalidInputException("vs_fixed: invalid validity byte");
		}
	}
}

void FixedFromBind(CopyFunction::CopyFromBindInput &input) {
	input.SetBindData<FixedReadBind>(FixedReadBind {input.GetFilePath(), BindFixed(input)});
}

void FixedFromInitLocal(CopyFunction::CopyFromInitLocalInput &input) {
	const auto &bind = input.GetBindData<FixedReadBind>();
	auto file = input.GetContext().GetFileSystem().OpenFile(bind.file_path, {FileFlags::READ});
	const auto size = file.Size();
	const auto expected = FixedHeader(bind.fixed.columns);
	std::string header(expected.size(), '\0');
	if (size < header.size()) {
		throw InvalidInputException("vs_fixed: the file is too short for its header");
	}
	file.ReadAt(header.data(), header.size(), 0);
	if (header != expected) {
		throw InvalidInputException("vs_fixed: the file's columns do not match the target table");
	}
	const auto row_width = bind.fixed.columns.size() * FIXED_VALUE_WIDTH;
	const auto data_size = size - header.size();
	if (data_size % row_width != 0) {
		throw InvalidInputException("vs_fixed: the file ends in a partial row");
	}
	input.SetLocalState<FixedReadState>(std::move(file), header.size(), data_size / row_width);
}

void FixedFromExec(CopyFunction::CopyFromExecInput &input) {
	const auto &bind = input.GetBindData<FixedReadBind>().fixed;
	auto &state = input.GetLocalState<FixedReadState>();
	auto output = input.GetOutputChunk();
	const auto rows = std::min(state.rows_left, output.GetCapacity());
	if (rows == 0) {
		return;
	}
	const auto size = rows * bind.columns.size() * FIXED_VALUE_WIDTH;
	state.buffer.resize(size);
	state.file.ReadAt(state.buffer.data(), size, state.offset);
	state.offset += size;
	state.rows_left -= rows;
	for (idx_t index = 0; index < bind.columns.size(); index++) {
		switch (bind.columns[index]) {
		case FixedColumn::BIGINT:
			DecodeFixedColumn<int64_t>(output, index, rows, state.buffer);
			break;
		case FixedColumn::DOUBLE:
			DecodeFixedColumn<double>(output, index, rows, state.buffer);
			break;
		}
	}
}

void RegisterLines(Connection &conn) {
	auto function = CopyFunction::Create(conn.GetFactory());
	function.SetName("vs_lines")
	    .SetCopyToBindCallback(LinesToBind)
	    .SetCopyToInitCallback(LinesToInit)
	    .SetCopyToBatchCallback(LinesToBatch)
	    .SetCopyToFlushCallback(LinesToFlush)
	    .SetCopyToFinalizeCallback(LinesToFinalize)
	    .SetCopyFromBindCallback(LinesFromBind)
	    .SetCopyFromInitLocalCallback(LinesFromInitLocal)
	    .SetCopyFromExecCallback(LinesFromExec);
	conn.Register(function);
}

void RegisterFixed(Connection &conn) {
	auto function = CopyFunction::Create(conn.GetFactory());
	function.SetName("vs_fixed")
	    .SetCopyToBindCallback(FixedToBind)
	    .SetCopyToInitCallback(FixedToInit)
	    .SetCopyToBatchCallback(FixedToBatch)
	    .SetCopyToFlushCallback(FixedToFlush)
	    .SetCopyToFinalizeCallback(FixedToFinalize)
	    .SetCopyFromBindCallback(FixedFromBind)
	    .SetCopyFromInitLocalCallback(FixedFromInitLocal)
	    .SetCopyFromExecCallback(FixedFromExec);
	conn.Register(function);
}

} // namespace

VERSUS_FAMILY(copy) {
	RegisterLines(conn);
	RegisterFixed(conn);
}
