#[path = "test/mod.rs"]
mod testm;

#[path = "tools/mod.rs"]
mod toolsm;

#[pyo3::pymodule]
mod _impl
{
    #[pymodule_export]
    use super::test;

    #[pymodule_export]
    use super::tools;
}

#[pyo3::pymodule]
mod test
{
    #[pyo3::pymodule]
    mod sass
    {
        #[pyo3::pymodule]
        mod _instruction
        {
            #[pymodule_export]
            use super::super::super::testm::sass::instruction::instruction::parse_any;
        }
    }
}

#[pyo3::pymodule]
mod tools
{
    #[pyo3::pymodule]
    mod sass
    {
        #[pyo3::pymodule]
        mod _decode
        {
            #[pymodule_export]
            use super::super::super::toolsm::sass::decode::normalize;

            #[pymodule_export]
            use super::super::super::toolsm::sass::decode::parse_instruction_and_controlcode;
        }
    }
}
