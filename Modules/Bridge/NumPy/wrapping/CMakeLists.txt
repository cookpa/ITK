if(ITK_WRAP_PYTHON)
  itk_wrap_module(ITKBridgeNumPy)
  set(
    WRAPPER_SUBMODULE_ORDER
    itkPyBuffer
    itkPyVectorContainer
  )
  set(
    WRAPPER_SWIG_LIBRARY_FILES
    ${WRAPPER_SWIG_LIBRARY_FILES}
    "${CMAKE_CURRENT_BINARY_DIR}/PyBuffer.i"
  )
  itk_auto_load_and_end_wrap_submodules()
endif()
