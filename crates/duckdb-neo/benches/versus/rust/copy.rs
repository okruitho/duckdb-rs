//! The `copy` family on duckdb-neo; the C++ side is `../cpp/copy.cpp`.
//!
//! `vs_lines`: text, one row per line ending in `\n`, fields separated by the `sep` option (one ASCII
//! punctuation or whitespace character other than `\`, `-`, `+` and `.`; default tab). NULL is `\N`;
//! VARCHAR escapes `\` as `\\`, newline as `\n` and the separator as `\` followed by it. BIGINT is
//! decimal, BOOLEAN `true`/`false`, DOUBLE scientific with 17 significant digits (`1.5000000000000000e0`,
//! `-1.0000000000000001e-7`, `nan`, `inf`, `-inf`): exact, so both standard libraries round it alike, where
//! their shortest round-trip forms break ties differently.
//! `vs_fixed`: binary BIGINT/DOUBLE: `VSFX`, the column count as a little-endian u32 and a type byte per
//! column (0 BIGINT, 1 DOUBLE), then per row and column a validity byte (1 valid, 0 NULL) and the value
//! as 8 little-endian bytes (zero for NULL).
//! COPY TO encodes each batch in the batch callback and writes it in flush. COPY FROM reads `vs_lines`
//! through a 1 MiB buffer, doubled for longer lines, and `vs_fixed` with one read per output chunk.

use std::{io::Write, ops::Range, sync::Mutex};

use duckdb_neo::{
    Error, Result,
    column_data_collection::ColumnDataCollection,
    connection::{Connection, Context},
    copy_function::{
        CopyFromBindInfo, CopyFromFunctionCallbacks, CopyFunctionBuilder, CopyToBindInfo, CopyToFunctionCallbacks,
    },
    data_chunk::DataChunk,
    file::{File, FileBuilder, FileSystem},
    logical_type::{LogicalType, LogicalTypeID},
    types::UncheckedString,
    value::Value,
    vector::{NativeElement, VectorReader, WritableNativeElement},
};

const LINES_BUFFER_SIZE: usize = 1 << 20;
const NULL_FIELD: &[u8] = b"\\N";
const FIXED_MAGIC: &[u8; 4] = b"VSFX";
const FIXED_VALUE_WIDTH: usize = 9;

/// One output file of `COPY ... TO`.
struct CopyFile {
    file: Mutex<File>,
}

fn open_output(context: &Context, file_path: &str) -> Result<File> {
    let fs = FileSystem::new(context)?;
    FileBuilder::new(&fs, file_path)?.write().create_new().open()
}

fn write_all(file: &File, mut bytes: &[u8]) -> Result<()> {
    while !bytes.is_empty() {
        let written = file.write(bytes)?;
        if written == 0 {
            return Err(Error::invalid_input("copy output file accepted no bytes"));
        }
        bytes = &bytes[written..];
    }
    Ok(())
}

fn invalid_field(format: &str, type_name: &str, field: &[u8]) -> Error {
    Error::invalid_input(format!(
        "{format}: invalid {type_name} field '{}'",
        String::from_utf8_lossy(field)
    ))
}

// ---------------------------------------------------------------------------------------------------------------------
// vs_lines
// ---------------------------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum LinesColumn {
    BigInt,
    Double,
    Boolean,
    Varchar,
}

impl LinesColumn {
    fn from_logical_type(logical_type: &LogicalType) -> Result<Self> {
        Ok(match logical_type.id()? {
            LogicalTypeID::BigInt => Self::BigInt,
            LogicalTypeID::Double => Self::Double,
            LogicalTypeID::Boolean => Self::Boolean,
            LogicalTypeID::Varchar => Self::Varchar,
            _ => {
                return Err(Error::invalid_input(format!(
                    "vs_lines does not support {}",
                    logical_type.to_text()?
                )));
            }
        })
    }
}

/// The separator and column types of a `vs_lines` file.
struct LinesBind {
    sep: u8,
    columns: Vec<LinesColumn>,
}

fn is_lines_sep(byte: u8) -> bool {
    (byte.is_ascii_punctuation() && !matches!(byte, b'\\' | b'-' | b'+' | b'.')) || byte == b' ' || byte == b'\t'
}

fn lines_sep(value: &Value) -> Result<u8> {
    let text = if value.logical_type()?.id()? == LogicalTypeID::Varchar {
        value.get::<String>()?
    } else {
        None
    };
    match text.as_deref().map(str::as_bytes) {
        Some(&[sep]) if is_lines_sep(sep) => Ok(sep),
        _ => Err(Error::invalid_input(
            "vs_lines: sep must be one ASCII punctuation or whitespace character other than \\, -, + and .",
        )),
    }
}

fn lines_bind<'a>(
    options: impl Iterator<Item = Result<(&'a str, Value)>>,
    types: impl Iterator<Item = Result<LogicalType>>,
) -> Result<LinesBind> {
    let mut sep = b'\t';
    for option in options {
        let (name, value) = option?;
        if !name.eq_ignore_ascii_case("sep") {
            return Err(Error::invalid_input(format!("vs_lines: unknown option {name}")));
        }
        sep = lines_sep(&value)?;
    }
    let columns = types
        .map(|logical_type| LinesColumn::from_logical_type(&logical_type?))
        .collect::<Result<_>>()?;
    Ok(LinesBind { sep, columns })
}

fn push_i64(value: i64, out: &mut Vec<u8>) {
    let mut digits = [0u8; 20];
    let mut start = digits.len();
    let mut rest = value.unsigned_abs();
    loop {
        start -= 1;
        digits[start] = b'0' + (rest % 10) as u8;
        rest /= 10;
        if rest == 0 {
            break;
        }
    }
    if value < 0 {
        out.push(b'-');
    }
    out.extend_from_slice(&digits[start..]);
}

fn push_f64(value: f64, out: &mut Vec<u8>) {
    if value.is_nan() {
        out.extend_from_slice(b"nan");
    } else if value.is_infinite() {
        out.extend_from_slice(if value < 0.0 { b"-inf" } else { b"inf" });
    } else {
        write!(out, "{value:.16e}").expect("writing to a Vec cannot fail");
    }
}

fn push_escaped(text: &[u8], sep: u8, out: &mut Vec<u8>) {
    for &byte in text {
        if byte == b'\\' {
            out.extend_from_slice(b"\\\\");
        } else if byte == b'\n' {
            out.extend_from_slice(b"\\n");
        } else if byte == sep {
            out.extend_from_slice(&[b'\\', sep]);
        } else {
            out.push(byte);
        }
    }
}

enum LinesReader<'a> {
    BigInt(VectorReader<'a, i64>),
    Double(VectorReader<'a, f64>),
    Boolean(VectorReader<'a, bool>),
    Varchar(VectorReader<'a, UncheckedString>),
}

impl LinesReader<'_> {
    fn encode_field(&self, row: usize, sep: u8, out: &mut Vec<u8>) -> Result<()> {
        let encoded = match self {
            Self::BigInt(reader) => reader.get(row)?.map(|value| push_i64(*value, out)),
            Self::Double(reader) => reader.get(row)?.map(|value| push_f64(*value, out)),
            Self::Boolean(reader) => reader
                .get(row)?
                .map(|value| out.extend_from_slice(if *value { b"true" } else { b"false" })),
            Self::Varchar(reader) => reader.get(row)?.map(|value| push_escaped(value.as_bytes(), sep, out)),
        };
        if encoded.is_none() {
            out.extend_from_slice(NULL_FIELD);
        }
        Ok(())
    }
}

fn encode_lines_chunk(bind: &LinesBind, chunk: &DataChunk<'_>, out: &mut Vec<u8>) -> Result<()> {
    let readers = bind
        .columns
        .iter()
        .enumerate()
        .map(|(index, column)| {
            Ok(match column {
                LinesColumn::BigInt => LinesReader::BigInt(chunk.get_vector_at(index)?.reader()?),
                LinesColumn::Double => LinesReader::Double(chunk.get_vector_at(index)?.reader()?),
                LinesColumn::Boolean => LinesReader::Boolean(chunk.get_vector_at(index)?.reader()?),
                LinesColumn::Varchar => LinesReader::Varchar(chunk.get_vector_at(index)?.reader()?),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    for row in 0..chunk.row_count()? {
        for (index, reader) in readers.iter().enumerate() {
            if index > 0 {
                out.push(bind.sep);
            }
            reader.encode_field(row, bind.sep, out)?;
        }
        out.push(b'\n');
    }
    Ok(())
}

/// The `vs_lines` text format.
struct VsLines;

impl CopyToFunctionCallbacks for VsLines {
    type InitData = CopyFile;
    type BindData = LinesBind;
    type BatchData = Vec<u8>;

    fn bind(&self, _context: &Context, bind_info: &CopyToBindInfo) -> Result<Self::BindData> {
        lines_bind(
            (0..bind_info.option_count()?).map(|index| bind_info.get_option(index)),
            (0..bind_info.column_count()?).map(|index| Ok(bind_info.get_column(index)?.1)),
        )
    }

    fn init(&self, context: &Context, _bind: &Self::BindData, file_path: &str) -> Result<Self::InitData> {
        Ok(CopyFile {
            file: Mutex::new(open_output(context, file_path)?),
        })
    }

    fn batch(
        &self,
        context: &Context,
        bind: &Self::BindData,
        _output: &Self::InitData,
        input: &ColumnDataCollection<'_>,
    ) -> Result<Self::BatchData> {
        let mut encoded = Vec::new();
        let mut scan = input.to_scan(context)?;
        while let Some(chunk) = scan.next_chunk()? {
            encode_lines_chunk(bind, chunk, &mut encoded)?;
        }
        Ok(encoded)
    }

    fn flush(
        &self,
        _context: &Context,
        _bind: &Self::BindData,
        output: &Self::InitData,
        _input: &ColumnDataCollection<'_>,
        encoded: &mut Self::BatchData,
    ) -> Result<()> {
        write_all(&output.file.lock().unwrap(), encoded)
    }

    fn finalize(&self, _context: &Context, _bind: &Self::BindData, output: &Self::InitData) -> Result<()> {
        output.file.lock().unwrap().close()
    }
}

/// The file path and layout of a `vs_lines` read.
struct LinesReadBind {
    file_path: String,
    lines: LinesBind,
}

/// A `vs_lines` read: the unparsed bytes `begin..end` of `buffer`, and the field ranges of the current chunk.
struct LinesReadState {
    file: File,
    buffer: Vec<u8>,
    begin: usize,
    end: usize,
    eof: bool,
    fields: Vec<Range<usize>>,
    unescaped: Vec<u8>,
}

impl LinesReadState {
    /// Read more of the file behind the unparsed bytes; false at the end of the file.
    fn refill(&mut self) -> Result<bool> {
        if self.eof {
            if self.begin < self.end {
                return Err(Error::invalid_input("vs_lines: the file does not end with a newline"));
            }
            return Ok(false);
        }
        self.buffer.copy_within(self.begin..self.end, 0);
        self.end -= self.begin;
        self.begin = 0;
        if self.end == self.buffer.len() {
            self.buffer.resize(self.buffer.len() * 2, 0);
        }
        let read = self.file.read_into(&mut self.buffer[self.end..])?;
        self.eof = read == 0;
        self.end += read;
        Ok(true)
    }

    /// Split up to `capacity` complete lines into fields; refills only before the first one.
    fn split_rows(&mut self, capacity: usize, bind: &LinesBind) -> Result<usize> {
        self.fields.clear();
        let mut rows = 0;
        while rows < capacity {
            match memchr::memchr(b'\n', &self.buffer[self.begin..self.end]) {
                Some(offset) => {
                    let line_end = self.begin + offset;
                    split_line(
                        &self.buffer,
                        self.begin..line_end,
                        bind.sep,
                        bind.columns.len(),
                        &mut self.fields,
                    )?;
                    self.begin = line_end + 1;
                    rows += 1;
                }
                None if rows == 0 => {
                    if !self.refill()? {
                        break;
                    }
                }
                None => break,
            }
        }
        Ok(rows)
    }
}

fn split_line(data: &[u8], line: Range<usize>, sep: u8, columns: usize, fields: &mut Vec<Range<usize>>) -> Result<()> {
    let mut field_start = line.start;
    let mut count = 1;
    let mut bytes = data[line.clone()].iter().zip(line.clone());
    while let Some((&byte, pos)) = bytes.next() {
        if byte == b'\\' {
            bytes.next();
        } else if byte == sep {
            fields.push(field_start..pos);
            count += 1;
            field_start = pos + 1;
        }
    }
    fields.push(field_start..line.end);
    if count != columns {
        return Err(Error::invalid_input(format!(
            "vs_lines: a row has {count} fields, expected {columns}"
        )));
    }
    Ok(())
}

fn parse_i64(field: &[u8]) -> Option<i64> {
    let (negative, digits) = match field {
        [b'-', rest @ ..] => (true, rest),
        _ => (false, field),
    };
    if digits.is_empty() || digits.len() > 19 {
        return None;
    }
    let mut value: u64 = 0;
    for &digit in digits {
        if !digit.is_ascii_digit() {
            return None;
        }
        value = value * 10 + u64::from(digit - b'0');
    }
    let limit = if negative { 1u64 << 63 } else { i64::MAX as u64 };
    if value > limit {
        return None;
    }
    Some(if negative { 0u64.wrapping_sub(value) } else { value } as i64)
}

fn parse_f64(field: &[u8]) -> Option<f64> {
    let text = std::str::from_utf8(field).ok()?;
    if text.starts_with('+') {
        return None;
    }
    text.parse().ok()
}

fn parse_bool(field: &[u8]) -> Option<bool> {
    match field {
        b"true" => Some(true),
        b"false" => Some(false),
        _ => None,
    }
}

/// The field with its escapes resolved; the field itself when it has none, which saves the copy.
fn unescape<'a>(field: &'a [u8], sep: u8, out: &'a mut Vec<u8>) -> Result<&'a [u8]> {
    if memchr::memchr(b'\\', field).is_none() {
        return Ok(field);
    }
    out.clear();
    let mut bytes = field.iter();
    while let Some(&byte) = bytes.next() {
        if byte != b'\\' {
            out.push(byte);
            continue;
        }
        match bytes.next() {
            Some(b'\\') => out.push(b'\\'),
            Some(b'n') => out.push(b'\n'),
            Some(&escaped) if escaped == sep => out.push(sep),
            _ => return Err(invalid_field("vs_lines", "VARCHAR", field)),
        }
    }
    Ok(out)
}

fn decode_lines_column<'f, T>(
    output: &mut DataChunk<'_>,
    index: usize,
    rows: usize,
    fields: impl Iterator<Item = &'f [u8]>,
    type_name: &str,
    parse: impl Fn(&[u8]) -> Option<T>,
) -> Result<()>
where
    T: NativeElement + WritableNativeElement<Write = T>,
{
    let mut vector = output.get_vector_at_mut::<T>(index)?;
    vector.set_size(rows)?;
    let mut writer = vector.writer()?;
    for (row, field) in fields.enumerate() {
        if field == NULL_FIELD {
            writer.set_null(row)?;
        } else {
            let value = parse(field).ok_or_else(|| invalid_field("vs_lines", type_name, field))?;
            writer.set(row, &value)?;
        }
    }
    Ok(())
}

fn decode_lines_varchar<'f>(
    output: &mut DataChunk<'_>,
    index: usize,
    rows: usize,
    fields: impl Iterator<Item = &'f [u8]>,
    sep: u8,
    unescaped: &mut Vec<u8>,
) -> Result<()> {
    let mut vector = output.get_vector_at_mut::<String>(index)?;
    vector.set_size(rows)?;
    let mut writer = vector.writer()?;
    for (row, field) in fields.enumerate() {
        if field == NULL_FIELD {
            writer.set_null(row)?;
            continue;
        }
        let text = std::str::from_utf8(unescape(field, sep, unescaped)?)
            .map_err(|_| invalid_field("vs_lines", "VARCHAR", field))?;
        writer.set(row, text)?;
    }
    Ok(())
}

impl CopyFromFunctionCallbacks for VsLines {
    type BindData = LinesReadBind;
    type LocalState = LinesReadState;
    type GlobalState = ();

    fn bind(&self, _context: &Context, file_path: &str, bind_info: &CopyFromBindInfo) -> Result<Self::BindData> {
        let lines = lines_bind(
            (0..bind_info.option_count()?).map(|index| bind_info.get_option(index)),
            (0..bind_info.column_count()?).map(|index| Ok(bind_info.get_column(index)?.1)),
        )?;
        Ok(LinesReadBind {
            file_path: file_path.to_owned(),
            lines,
        })
    }

    fn init_local_state(
        &self,
        context: &Context,
        bind: &Self::BindData,
        _global: Option<&Self::GlobalState>,
    ) -> Result<Option<Self::LocalState>> {
        let fs = FileSystem::new(context)?;
        let file = FileBuilder::new(&fs, &bind.file_path)?.read().open()?;
        Ok(Some(LinesReadState {
            file,
            buffer: vec![0; LINES_BUFFER_SIZE],
            begin: 0,
            end: 0,
            eof: false,
            fields: Vec::new(),
            unescaped: Vec::new(),
        }))
    }

    fn exec(
        &self,
        _context: &Context,
        bind: &Self::BindData,
        state: Option<&mut Self::LocalState>,
        _global: Option<&Self::GlobalState>,
        output: &mut DataChunk<'_>,
    ) -> Result<()> {
        let state = state.ok_or_else(|| Error::api_error("vs_lines: no read state"))?;
        let bind = &bind.lines;
        let rows = state.split_rows(output.capacity()?, bind)?;
        if rows == 0 {
            return Ok(());
        }
        let columns = bind.columns.len();
        for (index, column) in bind.columns.iter().enumerate() {
            let buffer = &state.buffer;
            let fields = state.fields[index..]
                .iter()
                .step_by(columns)
                .map(|range| &buffer[range.clone()]);
            match column {
                LinesColumn::BigInt => decode_lines_column(output, index, rows, fields, "BIGINT", parse_i64)?,
                LinesColumn::Double => decode_lines_column(output, index, rows, fields, "DOUBLE", parse_f64)?,
                LinesColumn::Boolean => decode_lines_column(output, index, rows, fields, "BOOLEAN", parse_bool)?,
                LinesColumn::Varchar => {
                    decode_lines_varchar(output, index, rows, fields, bind.sep, &mut state.unescaped)?;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// vs_fixed
// ---------------------------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FixedColumn {
    BigInt = 0,
    Double = 1,
}

impl FixedColumn {
    fn from_logical_type(logical_type: &LogicalType) -> Result<Self> {
        Ok(match logical_type.id()? {
            LogicalTypeID::BigInt => Self::BigInt,
            LogicalTypeID::Double => Self::Double,
            _ => {
                return Err(Error::invalid_input(format!(
                    "vs_fixed does not support {}",
                    logical_type.to_text()?
                )));
            }
        })
    }
}

fn fixed_columns(option_count: usize, types: impl Iterator<Item = Result<LogicalType>>) -> Result<Vec<FixedColumn>> {
    if option_count > 0 {
        return Err(Error::invalid_input("vs_fixed takes no options"));
    }
    types
        .map(|logical_type| FixedColumn::from_logical_type(&logical_type?))
        .collect()
}

fn fixed_header(columns: &[FixedColumn]) -> Vec<u8> {
    let mut header = FIXED_MAGIC.to_vec();
    header.extend_from_slice(&(columns.len() as u32).to_le_bytes());
    header.extend(columns.iter().map(|column| *column as u8));
    header
}

/// Fill one column of the rows in `out`, whose NULL values are already zero.
fn encode_fixed_column<'a, T: Copy + 'a>(
    values: impl Iterator<Item = Option<&'a T>>,
    to_le_bytes: impl Fn(T) -> [u8; 8],
    offset: usize,
    row_width: usize,
    out: &mut [u8],
) {
    for (row, value) in out.chunks_exact_mut(row_width).zip(values) {
        if let Some(value) = value {
            let field = &mut row[offset..offset + FIXED_VALUE_WIDTH];
            field[0] = 1;
            field[1..].copy_from_slice(&to_le_bytes(*value));
        }
    }
}

// The row width is fixed, so the chunk's rows are sized up front and filled a column at a time.
fn encode_fixed_chunk(columns: &[FixedColumn], chunk: &DataChunk<'_>, out: &mut Vec<u8>) -> Result<()> {
    let row_width = columns.len() * FIXED_VALUE_WIDTH;
    let start = out.len();
    out.resize(start + chunk.row_count()? * row_width, 0);
    let out = &mut out[start..];
    for (index, column) in columns.iter().enumerate() {
        let offset = index * FIXED_VALUE_WIDTH;
        match column {
            FixedColumn::BigInt => {
                let reader = chunk.get_vector_at::<i64>(index)?.reader()?;
                encode_fixed_column(reader.iter(), i64::to_le_bytes, offset, row_width, out);
            }
            FixedColumn::Double => {
                let reader = chunk.get_vector_at::<f64>(index)?.reader()?;
                encode_fixed_column(reader.iter(), f64::to_le_bytes, offset, row_width, out);
            }
        }
    }
    Ok(())
}

/// The `vs_fixed` binary format.
struct VsFixed;

impl CopyToFunctionCallbacks for VsFixed {
    type InitData = CopyFile;
    type BindData = Vec<FixedColumn>;
    type BatchData = Vec<u8>;

    fn bind(&self, _context: &Context, bind_info: &CopyToBindInfo) -> Result<Self::BindData> {
        fixed_columns(
            bind_info.option_count()?,
            (0..bind_info.column_count()?).map(|index| Ok(bind_info.get_column(index)?.1)),
        )
    }

    fn init(&self, context: &Context, columns: &Self::BindData, file_path: &str) -> Result<Self::InitData> {
        let file = open_output(context, file_path)?;
        write_all(&file, &fixed_header(columns))?;
        Ok(CopyFile { file: Mutex::new(file) })
    }

    fn batch(
        &self,
        context: &Context,
        columns: &Self::BindData,
        _output: &Self::InitData,
        input: &ColumnDataCollection<'_>,
    ) -> Result<Self::BatchData> {
        let mut encoded = Vec::new();
        let mut scan = input.to_scan(context)?;
        while let Some(chunk) = scan.next_chunk()? {
            encode_fixed_chunk(columns, chunk, &mut encoded)?;
        }
        Ok(encoded)
    }

    fn flush(
        &self,
        _context: &Context,
        _columns: &Self::BindData,
        output: &Self::InitData,
        _input: &ColumnDataCollection<'_>,
        encoded: &mut Self::BatchData,
    ) -> Result<()> {
        write_all(&output.file.lock().unwrap(), encoded)
    }

    fn finalize(&self, _context: &Context, _columns: &Self::BindData, output: &Self::InitData) -> Result<()> {
        output.file.lock().unwrap().close()
    }
}

/// The file path and column types of a `vs_fixed` read.
struct FixedReadBind {
    file_path: String,
    columns: Vec<FixedColumn>,
}

/// A `vs_fixed` read: the offset and number of the rows not read yet, and the current chunk's bytes.
struct FixedReadState {
    file: File,
    offset: usize,
    rows_left: usize,
    buffer: Vec<u8>,
}

fn decode_fixed_column<T>(
    output: &mut DataChunk<'_>,
    index: usize,
    rows: usize,
    data: &[u8],
    from_le_bytes: impl Fn([u8; 8]) -> T,
) -> Result<()>
where
    T: NativeElement + WritableNativeElement<Write = T>,
{
    let row_width = data.len() / rows;
    let offset = index * FIXED_VALUE_WIDTH;
    let mut vector = output.get_vector_at_mut::<T>(index)?;
    vector.set_size(rows)?;
    let mut writer = vector.writer()?;
    for (row, bytes) in data.chunks_exact(row_width).enumerate() {
        let field = &bytes[offset..offset + FIXED_VALUE_WIDTH];
        let value: [u8; 8] = field[1..].try_into().expect("a fixed value has 8 value bytes");
        match field[0] {
            0 => writer.set_null(row)?,
            1 => writer.set(row, &from_le_bytes(value))?,
            _ => return Err(Error::invalid_input("vs_fixed: invalid validity byte")),
        }
    }
    Ok(())
}

impl CopyFromFunctionCallbacks for VsFixed {
    type BindData = FixedReadBind;
    type LocalState = FixedReadState;
    type GlobalState = ();

    fn bind(&self, _context: &Context, file_path: &str, bind_info: &CopyFromBindInfo) -> Result<Self::BindData> {
        let columns = fixed_columns(
            bind_info.option_count()?,
            (0..bind_info.column_count()?).map(|index| Ok(bind_info.get_column(index)?.1)),
        )?;
        Ok(FixedReadBind {
            file_path: file_path.to_owned(),
            columns,
        })
    }

    fn init_local_state(
        &self,
        context: &Context,
        bind: &Self::BindData,
        _global: Option<&Self::GlobalState>,
    ) -> Result<Option<Self::LocalState>> {
        let fs = FileSystem::new(context)?;
        let file = FileBuilder::new(&fs, &bind.file_path)?.read().open()?;
        let size = file.size()? as usize;
        let expected = fixed_header(&bind.columns);
        let mut header = vec![0; expected.len()];
        if size < header.len() {
            return Err(Error::invalid_input("vs_fixed: the file is too short for its header"));
        }
        file.read_exact_at(0, &mut header)?;
        if header != expected {
            return Err(Error::invalid_input(
                "vs_fixed: the file's columns do not match the target table",
            ));
        }
        let row_width = bind.columns.len() * FIXED_VALUE_WIDTH;
        let data_size = size - header.len();
        if !data_size.is_multiple_of(row_width) {
            return Err(Error::invalid_input("vs_fixed: the file ends in a partial row"));
        }
        Ok(Some(FixedReadState {
            file,
            offset: header.len(),
            rows_left: data_size / row_width,
            buffer: Vec::new(),
        }))
    }

    fn exec(
        &self,
        _context: &Context,
        bind: &Self::BindData,
        state: Option<&mut Self::LocalState>,
        _global: Option<&Self::GlobalState>,
        output: &mut DataChunk<'_>,
    ) -> Result<()> {
        let state = state.ok_or_else(|| Error::api_error("vs_fixed: no read state"))?;
        let rows = state.rows_left.min(output.capacity()?);
        if rows == 0 {
            return Ok(());
        }
        let size = rows * bind.columns.len() * FIXED_VALUE_WIDTH;
        state.buffer.resize(size, 0);
        state.file.read_exact_at(state.offset, &mut state.buffer)?;
        state.offset += size;
        state.rows_left -= rows;
        for (index, column) in bind.columns.iter().enumerate() {
            match column {
                FixedColumn::BigInt => decode_fixed_column(output, index, rows, &state.buffer, i64::from_le_bytes)?,
                FixedColumn::Double => decode_fixed_column(output, index, rows, &state.buffer, f64::from_le_bytes)?,
            }
        }
        Ok(())
    }
}

pub fn register(conn: &Connection) -> Result<()> {
    CopyFunctionBuilder::new("vs_lines", VsLines).register_to_and_from(conn)?;
    CopyFunctionBuilder::new("vs_fixed", VsFixed).register_to_and_from(conn)
}
