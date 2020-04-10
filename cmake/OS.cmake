# Test Apple first because UNIX==true for Apple and Linux.
if (APPLE)
  set(OCCA_OS "OCCA_MACOS_OS")
elseif (UNIX)
  set(OCCA_OS "OCCA_LINUX_OS")
else()
  set(OCCA_OS "OCCA_WINDOWS_OS")
endif()
