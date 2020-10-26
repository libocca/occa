#include <iostream>
#include <sstream>
#include <occa/defines.hpp>
#include <occa/tools/cli.hpp>
#include <occa/tools/testing.hpp>


int main(const int argc, const char **argv) {
  //---[ OCCA ]----------------------------
  ASSERT_EQ(
    autocomplete("occa"),
    "autocomplete clear compile env info modes translate version"
  );

  ASSERT_EQ(
    autocomplete("occa a"),
    "autocomplete"
  );

  ASSERT_EQ(
    autocomplete("occa c"),
    "clear compile"
  );

  ASSERT_EQ(
    autocomplete("occa cl"),
    "cl"
  );

  ASSERT_EQ(
    autocomplete("occa -"),
    "--help"
  );

  ASSERT_EQ(
    autocomplete("occa --"),
    "--help"
  );

  //---[ Autocomplete ]--------------------
  ASSERT_EQ(
    autocomplete("occa autocomplete"),
    "bash"
  );

  ASSERT_EQ(
    autocomplete("occa autocomplete b"),
    "bash"
  );

  ASSERT_EQ(
    autocomplete("occa autocomplete bash"),
    ""
  );

  //---[ Clear ]---------------------------
  ASSERT_EQ(
    autocomplete("occa clear"),
    "--all --help --kernels --lib --libraries --locks --yes"
  );

  ASSERT_EQ(
    autocomplete("occa clear --a"),
    "--all"
  );

  ASSERT_EQ(
    autocomplete("occa clear --all"),
    ""
  );

  ASSERT_EQ(
    autocomplete("occa clear --l"),
    "--lib --libraries --locks"
  );

  ASSERT_EQ(
    autocomplete("occa clear --li"),
    "--lib --libraries"
  );

  ASSERT_EQ(
    autocomplete("occa clear --lib"),
    "lib1_1 lib1_2 lib2"
  );

  ASSERT_EQ(
    autocomplete("occa clear --lib lib1"),
    "lib1_1 lib1_2"
  );

  ASSERT_EQ(
    autocomplete("occa clear --locks"),
    ""
  );

  ASSERT_EQ(
    autocomplete("occa clear --locks --l"),
    "--lib --libraries"
  );

  ASSERT_EQ(
    autocomplete("occa clear --locks --libraries --l"),
    "--lib"
  );

  ASSERT_EQ(
    autocomplete("occa clear --locks --libraries --lib foo --l"),
    "--lib"
  );

  //---[ Env ]-----------------------------
  ASSERT_EQ(
    autocomplete("occa env"),
    "--help"
  );

  ASSERT_EQ(
    autocomplete("occa env -"),
    "--help"
  );


  ASSERT_EQ(
    autocomplete("occa env --help"),
    ""
  );

  //---[ Info ]----------------------------
  ASSERT_EQ(
    autocomplete("occa info"),
    "--help"
  );

  ASSERT_EQ(
    autocomplete("occa info -"),
    "--help"
  );


  ASSERT_EQ(
    autocomplete("occa info --help"),
    ""
  );

  //---[ Kernel ]--------------------------
  ASSERT_EQ(
    autocomplete("occa kernel"),
    "compile info translate"
  );

  ASSERT_EQ(
    autocomplete("occa kernel c"),
    "compile"
  );

  ASSERT_EQ(
    autocomplete("occa kernel t"),
    "translate"
  );

  ASSERT_EQ(
    autocomplete("occa kernel -"),
    "--help"
  );

  ASSERT_EQ(
    autocomplete("occa kernel --"),
    "--help"
  );

  //---[ Kernel Compile ]------------------
  ASSERT_EQ(
    autocomplete("occa kernel compile"),
    "dir1/ file1 file2"
  );

  ASSERT_EQ(
    autocomplete("occa kernel compile dir1/"),
    "file3 file4"
  );

  ASSERT_EQ(
    autocomplete("occa kernel compile file1"),
    "--device-props --help --kernel-props --include-path --define"
  );

  ASSERT_EQ(
    autocomplete("occa kernel compile file1"),
    "--device-props --help --kernel-props --include-path --define"
  );

  ASSERT_EQ(
    autocomplete("occa kernel compile file1"),
    "--device-props --help --kernel-props --include-path --define"
  );

  ASSERT_EQ(
    autocomplete("occa kernel compile file1"),
    "--device-props --help --kernel-props --include-path --define"
  );

  ASSERT_EQ(
    autocomplete("occa kernel compile --device-props"),
    "dir1/ file1 file2"
  );

  ASSERT_EQ(
    autocomplete("occa kernel compile --kernel-props"),
    "dir1/ file1 file2"
  );

  //---[ Kernel Translate ]----------------
  ASSERT_EQ(
    autocomplete("occa kernel translate"),
    "dir1/ file1 file2"
  );

  ASSERT_EQ(
    autocomplete("occa kernel translate dir1/"),
    "file3 file4"
  );

  ASSERT_EQ(
    autocomplete("occa kernel translate file1"),
    "--define --include-path --kernel-props --launcher --mode --verbose"
  );

  ASSERT_EQ(
    autocomplete("occa kernel translate file1 --mode"),
    "Serial"
  );

  ASSERT_EQ(
    autocomplete("occa kernel translate file1 --mode S"),
    "Serial"
  );

  ASSERT_EQ(
    autocomplete("occa kernel translate file1 --mode Serial"),
    "--define --include-path --kernel-props --launcher --verbose"
  );

  ASSERT_EQ(
    autocomplete("occa kernel translate file1 --mode Serial --launcher"),
    "--define --include-path --kernel-props --verbose"
  );

  ASSERT_EQ(
    autocomplete("occa kernel translate file1 --mode Serial --launcher --verbose"),
    "--define --include-path --kernel-props"
  );

  ASSERT_EQ(
    autocomplete("occa kernel translate --kernel-props"),
    "dir1/ file1 file2"
  );

  //---[ Kernel Translate ]----------------
  ASSERT_EQ(
    autocomplete("occa kernel info"),
    "hash1 hash2 hash3"
  );

  ASSERT_EQ(
    autocomplete("occa kernel info hash"),
    "hash1 hash2 hash3"
  );

  ASSERT_EQ(
    autocomplete("occa kernel info hash1"),
    ""
  );

  //---[ Modes ]---------------------------
  ASSERT_EQ(
    autocomplete("occa modes"),
    "--help"
  );

  ASSERT_EQ(
    autocomplete("occa modes -"),
    "--help"
  );


  ASSERT_EQ(
    autocomplete("occa modes --help"),
    ""
  );

  // //---[ Version ]-------------------------
  ASSERT_EQ(
    autocomplete("occa version"),
    "--okl"
  );

  ASSERT_EQ(
    autocomplete("occa version --okl"),
    ""
  );

  return 0;
}
