target_sources(libocca PRIVATE attribute.cpp enumerator.cpp file.cpp
  kernelMetadata.cpp keyword.cpp macro.cpp operator.cpp parser.cpp
  preprocessor.cpp printer.cpp processingStages.cpp qualifier.cpp
  scope.cpp specialMacros.cpp statementContext.cpp statementPeeker.cpp
  tokenContext.cpp tokenizer.cpp variable.cpp)

add_subdirectory(builtins)
add_subdirectory(expr)
add_subdirectory(loaders)
add_subdirectory(modes)
add_subdirectory(statement)
add_subdirectory(token)
add_subdirectory(type)
