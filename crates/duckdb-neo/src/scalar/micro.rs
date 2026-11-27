use std::marker::PhantomData;

use super::{ScalarCallbacks, ScalarFunctionBuilder};
use crate::{
    DuckDBType, Error, Result,
    connection::Context,
    data_chunk::VectorCollection,
    enums::FunctionProperty,
    signature::{FunctionParameter, SignatureBuilder, TypeDeclaration},
    vector::{
        NativeElement, PlainNative, RowLayout, StorageKind, Unknown, Vector, VectorReader, WritableNativeElement,
    },
};

/// Creates scalar functions from vector closures.
///
/// [`Self::create`] infers a unary signature from the input and output element
/// types. [`Self::with_signature`] accepts an explicit signature and an input
/// collection for multiple arguments, varargs, or runtime logical types.
/// [`Self::unary`] through [`Self::quaternary`] take a function of one row
/// instead, for plain argument and result types.
/// The output vector is already sized to the batch's row count; the closure
/// must write every output row. DuckDB's default NULL handling applies unless
/// overridden with [`ScalarFunctionBuilder::set_property`] or, for the row
/// functions, requested by an `Option` argument.
///
/// Closures may capture owned `Send + Sync` state; binding and worker-local
/// initialization remain available through [`ScalarCallbacks`] instead.
///
/// ```
/// use duckdb_neo::{
///     Parameters,
///     environment::{Environment, StorageLocation},
///     scalar::MicroScalar,
///     vector::Vector,
/// };
///
/// # fn main() -> duckdb_neo::Result<()> {
/// let function = MicroScalar::create(
///     "plus_one",
///     |_ctx, input: Vector<'_, i32>, mut output: Vector<'_, i64>| {
///         output.writer()?.map_from(&input.reader()?, |value| i64::from(*value) + 1)
///     },
/// );
/// let env = Environment::new()?;
/// let conn = env.open(StorageLocation::InMemory)?.connect()?;
/// function.register(&conn)?;
///
/// let chunk = conn.query("SELECT plus_one(41)", Parameters::None)?
///     .next_chunk()?.unwrap();
/// assert_eq!(chunk.get_vector_at::<i64>(0)?.reader()?.get(0)?, Some(&42));
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct MicroScalar;

impl MicroScalar {
    /// Infer a unary signature and wrap the closure in a scalar function builder.
    pub fn create<I, O, F>(name: impl Into<String>, function: F) -> ScalarFunctionBuilder<impl ScalarCallbacks>
    where
        I: DuckDBType + NativeElement + 'static,
        O: DuckDBType + WritableNativeElement + 'static,
        F: Fn(&Context, Vector<'_, I>, Vector<'_, O>) -> Result<()> + Send + Sync + 'static,
    {
        let mut signature = SignatureBuilder::new().param(FunctionParameter::new::<I>("input"));
        signature.return_type = Some(TypeDeclaration::rust_type::<O>());
        Self::with_signature(name, signature, move |context, input, output| {
            function(context, input.get_vector_at_mut::<I>(0)?, output)
        })
    }

    /// Wrap a closure with an explicit signature and checked output element type.
    /// Arguments follow the ordering of [`SignatureBuilder::param`].
    pub fn with_signature<O, F>(
        name: impl Into<String>,
        signature: SignatureBuilder,
        function: F,
    ) -> ScalarFunctionBuilder<impl ScalarCallbacks>
    where
        O: WritableNativeElement + 'static,
        F: Fn(&Context, &mut VectorCollection, Vector<'_, O>) -> Result<()> + Send + Sync + 'static,
    {
        ScalarFunctionBuilder::new(
            name,
            signature,
            TypedCallback::<O, F> {
                function,
                types: PhantomData,
            },
        )
    }

    /// Call `function` once per row of argument `arg0`, with the signature inferred from the types, like
    /// DuckDB's `ScalarExecutor`; see [`RowArgument`] and [`RowResult`] for NULLs and errors.
    ///
    /// ```
    /// use duckdb_neo::{Parameters, environment::{Environment, StorageLocation}, scalar::MicroScalar};
    ///
    /// # fn main() -> duckdb_neo::Result<()> {
    /// let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
    /// MicroScalar::unary("halve", |value: i64| (value % 2 == 0).then_some(value / 2)).register(&conn)?;
    ///
    /// let chunk = conn.query("SELECT halve(i) FROM range(4) t(i)", Parameters::None)?.next_chunk()?.unwrap();
    /// let rows = chunk.get_vector_at::<i64>(0)?.reader()?.iter().map(|v| v.copied()).collect::<Vec<_>>();
    /// assert_eq!(rows, [Some(0), None, Some(1), None]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn unary<A, R, F>(name: impl Into<String>, function: F) -> ScalarFunctionBuilder<impl ScalarCallbacks>
    where
        A: RowArgument,
        R: RowResult,
        F: Fn(A) -> R + Send + Sync + 'static,
    {
        Self::rows::<(A,), R, F>(name, &["arg0"], function)
    }

    /// As [`Self::unary`], for two arguments named `arg0` and `arg1`.
    ///
    /// ```
    /// use duckdb_neo::{Parameters, environment::{Environment, StorageLocation}, scalar::MicroScalar};
    ///
    /// # fn main() -> duckdb_neo::Result<()> {
    /// let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
    /// MicroScalar::binary("checked_div", |a: i64, b: i64| a.checked_div(b)).register(&conn)?;
    ///
    /// let chunk = conn.query("SELECT checked_div(7, 2), checked_div(7, 0)", Parameters::None)?.next_chunk()?.unwrap();
    /// assert_eq!(chunk.get_vector_at::<i64>(0)?.reader()?.get(0)?, Some(&3));
    /// assert_eq!(chunk.get_vector_at::<i64>(1)?.reader()?.get(0)?, None);
    /// # Ok(())
    /// # }
    /// ```
    pub fn binary<A, B, R, F>(name: impl Into<String>, function: F) -> ScalarFunctionBuilder<impl ScalarCallbacks>
    where
        A: RowArgument,
        B: RowArgument,
        R: RowResult,
        F: Fn(A, B) -> R + Send + Sync + 'static,
    {
        Self::rows::<(A, B), R, F>(name, &["arg0", "arg1"], function)
    }

    /// As [`Self::unary`], for three arguments named `arg0` to `arg2`.
    pub fn ternary<A, B, C, R, F>(name: impl Into<String>, function: F) -> ScalarFunctionBuilder<impl ScalarCallbacks>
    where
        A: RowArgument,
        B: RowArgument,
        C: RowArgument,
        R: RowResult,
        F: Fn(A, B, C) -> R + Send + Sync + 'static,
    {
        Self::rows::<(A, B, C), R, F>(name, &["arg0", "arg1", "arg2"], function)
    }

    /// As [`Self::unary`], for four arguments named `arg0` to `arg3`.
    pub fn quaternary<A, B, C, D, R, F>(
        name: impl Into<String>,
        function: F,
    ) -> ScalarFunctionBuilder<impl ScalarCallbacks>
    where
        A: RowArgument,
        B: RowArgument,
        C: RowArgument,
        D: RowArgument,
        R: RowResult,
        F: Fn(A, B, C, D) -> R + Send + Sync + 'static,
    {
        Self::rows::<(A, B, C, D), R, F>(name, &["arg0", "arg1", "arg2", "arg3"], function)
    }

    fn rows<Args: RowArguments, R: RowResult, F>(
        name: impl Into<String>,
        names: &[&str],
        function: F,
    ) -> ScalarFunctionBuilder<RowCallback<Args, R, F>>
    where
        RowCallback<Args, R, F>: ScalarCallbacks,
    {
        let mut signature = Args::parameters(names);
        signature.return_type = Some(TypeDeclaration::rust_type::<R::Element>());
        let builder = ScalarFunctionBuilder::new(
            name,
            signature,
            RowCallback {
                function,
                types: PhantomData,
            },
        );
        if Args::NULLABLE {
            builder.set_property(FunctionProperty::HasSpecialNullHandling(true))
        } else {
            builder
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// An argument of [`MicroScalar::unary`] to [`MicroScalar::quaternary`]: a NULL in a [`PlainNative`] makes the row
/// NULL without a call; `Option<T>` gets it as `None` and sets [`FunctionProperty::HasSpecialNullHandling`].
pub trait RowArgument: sealed::Sealed + 'static {
    /// The element type of the SQL parameter.
    type Element: PlainNative + DuckDBType + 'static;
    #[doc(hidden)]
    const NULLABLE: bool;
    #[doc(hidden)]
    fn from_row(value: Option<Self::Element>) -> Self;
}

/// The result of [`MicroScalar::unary`] to [`MicroScalar::quaternary`]: a [`PlainNative`] or `Option` of one,
/// whose `None` makes the row NULL, either possibly in a [`Result`] whose error fails the call.
pub trait RowResult: sealed::Sealed + 'static {
    /// The element type of the SQL result.
    type Element: PlainNative + DuckDBType + 'static;
    #[doc(hidden)]
    fn into_row(self) -> Result<Option<Self::Element>>;
}

impl<T: PlainNative + DuckDBType + 'static> sealed::Sealed for T {}

impl<T: PlainNative + DuckDBType + 'static> sealed::Sealed for Option<T> {}

impl<T: PlainNative + DuckDBType + 'static> RowArgument for T {
    type Element = T;
    const NULLABLE: bool = false;

    #[inline(always)]
    fn from_row(value: Option<T>) -> T {
        value.expect("NULL rows of plain arguments do not reach the row function")
    }
}

impl<T: PlainNative + DuckDBType + 'static> RowArgument for Option<T> {
    type Element = T;
    const NULLABLE: bool = true;

    #[inline(always)]
    fn from_row(value: Option<T>) -> Self {
        value
    }
}

impl<T: PlainNative + DuckDBType + 'static> sealed::Sealed for Result<T> {}

impl<T: PlainNative + DuckDBType + 'static> sealed::Sealed for Result<Option<T>> {}

impl<T: PlainNative + DuckDBType + 'static> RowResult for T {
    type Element = T;

    #[inline(always)]
    fn into_row(self) -> Result<Option<T>> {
        Ok(Some(self))
    }
}

impl<T: PlainNative + DuckDBType + 'static> RowResult for Option<T> {
    type Element = T;

    #[inline(always)]
    fn into_row(self) -> Result<Option<T>> {
        Ok(self)
    }
}

impl<T: PlainNative + DuckDBType + 'static> RowResult for Result<T> {
    type Element = T;

    #[inline(always)]
    fn into_row(self) -> Result<Option<T>> {
        self.map(Some)
    }
}

impl<T: PlainNative + DuckDBType + 'static> RowResult for Result<Option<T>> {
    type Element = T;

    #[inline(always)]
    fn into_row(self) -> Result<Option<T>> {
        self
    }
}

trait RowArguments {
    const NULLABLE: bool;

    fn parameters(names: &[&str]) -> SignatureBuilder;
}

struct RowCallback<Args, R, F> {
    function: F,
    types: PhantomData<fn() -> (Args, R)>,
}

struct Column<'v, A: RowArgument> {
    data: *const A::Element,
    layout: RowLayout,
    len: usize,
    _reader: PhantomData<&'v A::Element>,
}

impl<'v, A: RowArgument> Column<'v, A> {
    fn new(reader: &VectorReader<'v, A::Element>, rows: usize) -> Result<Self> {
        if reader.len() < rows {
            return Err(Error::invalid_input(format!(
                "argument has {} rows, expected {rows}",
                reader.len()
            )));
        }
        Ok(Self {
            data: reader.as_ptr().cast(),
            layout: reader.layout(),
            len: reader.len(),
            _reader: PhantomData,
        })
    }

    #[inline(always)]
    fn dense(&self) -> Option<Dense<A>> {
        match self.layout {
            RowLayout::Flat { validity } if validity.is_null() => Some(Dense::Flat(FlatLane(self.data))),
            RowLayout::Constant { is_null: false } if self.len != 0 => {
                // SAFETY: a valid constant with rows holds a valid `A::Element` in slot zero.
                let value = unsafe { *self.data };
                Some(Dense::Constant(ConstantLane(value)))
            }
            _ => None,
        }
    }

    /// # Safety
    /// As for [`RowLayout::validity_word`].
    #[inline(always)]
    unsafe fn validity_word(&self, start: usize, end: usize) -> u64 {
        if A::NULLABLE {
            u64::MAX
        } else {
            // SAFETY: the caller guarantees the rows.
            unsafe { self.layout.validity_word(start, end) }
        }
    }
}

trait Lane {
    type Argument;

    /// # Safety
    /// `row` must be a row of the lane; for a plain argument, one where it is valid.
    unsafe fn get(&self, row: usize) -> Self::Argument;
}

impl<A: RowArgument> Lane for Column<'_, A> {
    type Argument = A;

    #[inline(always)]
    unsafe fn get(&self, row: usize) -> A {
        // SAFETY: the caller guarantees the row; slots of valid rows hold valid `A::Element`s, by `PlainNative`.
        unsafe {
            if A::NULLABLE {
                A::from_row(self.layout.valid_physical(row).map(|physical| *self.data.add(physical)))
            } else {
                A::from_row(Some(*self.data.add(self.layout.physical(row))))
            }
        }
    }
}

enum Dense<A: RowArgument> {
    Flat(FlatLane<A>),
    Constant(ConstantLane<A>),
}

struct FlatLane<A: RowArgument>(*const A::Element);

impl<A: RowArgument> Lane for FlatLane<A> {
    type Argument = A;

    #[inline(always)]
    unsafe fn get(&self, row: usize) -> A {
        // SAFETY: the caller guarantees the row, and every row of a flat vector without a mask is valid.
        A::from_row(Some(unsafe { *self.0.add(row) }))
    }
}

struct ConstantLane<A: RowArgument>(A::Element);

impl<A: RowArgument> Lane for ConstantLane<A> {
    type Argument = A;

    #[inline(always)]
    unsafe fn get(&self, _row: usize) -> A {
        A::from_row(Some(self.0))
    }
}

fn flatten_unreadable(input: &mut VectorCollection) -> Result<()> {
    for index in 0..input.col_count() {
        if input.get_untyped_vector_at(index)?.storage_kind() == StorageKind::Other {
            input.get_untyped_vector_at_mut(index)?.flatten()?;
        }
    }
    Ok(())
}

impl<A: RowArgument> RowArguments for (A,) {
    const NULLABLE: bool = A::NULLABLE;

    fn parameters(names: &[&str]) -> SignatureBuilder {
        SignatureBuilder::new().param(FunctionParameter::new::<A::Element>(names[0]))
    }
}

impl<A, R, F> ScalarCallbacks for RowCallback<(A,), R, F>
where
    A: RowArgument,
    R: RowResult,
    F: Fn(A) -> R + Send + Sync + 'static,
{
    type BindData = ();
    type InitData = ();

    fn exec(
        &self,
        _context: &Context,
        _bind_data: Option<&Self::BindData>,
        _init_data: Option<&mut Self::InitData>,
        input: &mut VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        let rows = input.row_count()?;
        flatten_unreadable(input)?;
        let reader = input.get_vector_at::<A::Element>(0)?.reader()?;
        let column = Column::<A>::new(&reader, rows)?;
        let mut output = output.cast::<R::Element>()?;
        let mut writer = output.writer()?;
        let (function, data, layout) = (&self.function, column.data, column.layout);
        // SAFETY: `map_layout` writes or NULLs each of the `rows` rows, all of which the column holds; `read`
        // gets the slot of a valid row, which holds an initialized value of the slot type.
        unsafe {
            writer.resize_and_write(rows, move |writer| {
                writer.map_layout(
                    rows,
                    layout,
                    !A::NULLABLE,
                    |physical, _| *data.add(physical),
                    |value| function(A::from_row(value)).into_row(),
                )
            })
        }
    }
}

macro_rules! dense_lanes {
    ([] => $body:expr) => {
        $body
    };
    ([$column:ident $($rest:ident)*] => $body:expr) => {
        match $column {
            Dense::Flat(lane) => {
                let $column = lane;
                dense_lanes!([$($rest)*] => $body)
            }
            Dense::Constant(lane) => {
                let $column = lane;
                dense_lanes!([$($rest)*] => $body)
            }
        }
    };
}

macro_rules! row_callback {
    ($(($argument:ident, $column:ident, $index:tt)),+) => {
        impl<$($argument: RowArgument),+> RowArguments for ($($argument,)+) {
            const NULLABLE: bool = false $(|| $argument::NULLABLE)+;

            fn parameters(names: &[&str]) -> SignatureBuilder {
                SignatureBuilder::new()
                    $(.param(FunctionParameter::new::<$argument::Element>(names[$index])))+
            }
        }

        impl<$($argument,)+ R, F> ScalarCallbacks for RowCallback<($($argument,)+), R, F>
        where
            $($argument: RowArgument,)+
            R: RowResult,
            F: Fn($($argument),+) -> R + Send + Sync + 'static,
        {
            type BindData = ();
            type InitData = ();

            fn exec(
                &self,
                _context: &Context,
                _bind_data: Option<&Self::BindData>,
                _init_data: Option<&mut Self::InitData>,
                input: &mut VectorCollection,
                output: Vector<'_, Unknown>,
            ) -> Result<()> {
                let rows = input.row_count()?;
                flatten_unreadable(input)?;
                $(let $column = input.get_vector_at::<$argument::Element>($index)?.reader()?;)+
                $(let $column = Column::<$argument>::new(&$column, rows)?;)+
                let mut output = output.cast::<R::Element>()?;
                let mut writer = output.writer()?;
                let function = &self.function;
                let propagate_nulls = !(true $(&& $argument::NULLABLE)+);
                // SAFETY: `write_rows` writes or NULLs each of the `rows` rows, all of which every column holds. It
                // only reads rows whose plain arguments are all valid, and dense lanes have no NULL rows.
                unsafe {
                    writer.resize_and_write(rows, move |writer| match ($($column.dense(),)+) {
                        ($(Some($column),)+) => dense_lanes!([$($column)+] => writer.write_rows(
                            rows,
                            propagate_nulls,
                            |_, _| u64::MAX,
                            |row, _| function($($column.get(row)),+).into_row(),
                        )),
                        _ => writer.write_rows(
                            rows,
                            propagate_nulls,
                            |start, end| u64::MAX $(& $column.validity_word(start, end))+,
                            |row, _| function($($column.get(row)),+).into_row(),
                        ),
                    })
                }
            }
        }
    };
}

row_callback!((A, a, 0), (B, b, 1));
row_callback!((A, a, 0), (B, b, 1), (C, c, 2));
row_callback!((A, a, 0), (B, b, 1), (C, c, 2), (D, d, 3));

struct TypedCallback<O, F> {
    function: F,
    types: PhantomData<fn() -> O>,
}

impl<O, F> ScalarCallbacks for TypedCallback<O, F>
where
    O: WritableNativeElement + 'static,
    F: Fn(&Context, &mut VectorCollection, Vector<'_, O>) -> Result<()> + Send + Sync + 'static,
{
    type BindData = ();
    type InitData = ();

    fn exec(
        &self,
        context: &Context,
        _bind_data: Option<&Self::BindData>,
        _init_data: Option<&mut Self::InitData>,
        input: &mut VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        let mut output = output.cast::<O>()?;
        output.set_size(input.row_count()?)?;
        (self.function)(context, input, output)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
