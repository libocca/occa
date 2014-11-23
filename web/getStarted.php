<?php include($_SERVER['DOCUMENT_ROOT'] . '/main.php'); ?>
<?php addHeader('OCCA: Get Started') ?>

<?php absInclude("/menu.php"); ?>

<div id="id_body" class="getStarted">


  <h2 class="ui dividing header"> Linux </h2>
  <div class="dsm5 indent1">
    <h4 class="ui dividing header"> Downloading Source </h4>
    <div class="dsm5 indent1">
      sudo apt-get install git
      <br/>git clone https://github.com/tcew/OCCA2
      <br/>This last command makes a directory OCCA2 and pulls the OCCA source into the new OCCA2 directory
      <br/>Copy the path to the OCCA directory, you can see this by typing
      <br/>pwd
      <br/>on the terminal
    </div>

    <h4 class="ui dividing header"> Installation </h4>
    <div class="dsm5 indent1">
      Install make
      <br/>sudo apt-get install make
      <br/>Go to the OCCA2 directory
      <br/>We're going to save the directory in a variable OCCA_DIR by using
      <br/>export OCCA_DIR=/absolute/path/to/OCCA2 # For instance: export OCCA_DIR=/home/dsm5/gitRepos/OCCA2
      <br/>You'll need this when using OCCA, so it's handy to type the line above in your shell init script (like ~/.bashrc or ~/.profile)
      <br/>Now we'll compile with
      <br/>make -j 8 # Compile in parallel with 8 threads, feel free to reduce the number
      <br/>Although the OCCA API is ported to multiple languages, the back-end is mainly in C++.
           To specify the C++ compiler for compiling OCCA, you can use
      <br/>make -j 8 CXX="clang++" # Where g++ or icpc, for example, could be used instead of clang++
      <br/>Check [Specialized Options] for more options
    </div>

    <h4 class="ui dividing header"> Running addVectors </h4>
    <div class="dsm5 indent1">
    </div>
  </div>

  <h2 class="ui dividing header"> Mac OS X </h2>
  <div class="dsm5 indent1">
    <h4 class="ui dividing header"> Downloading Source </h4>
    <div class="dsm5 indent1">
      Download Git from their
      <a href="http://git-scm.com/download/mac" class="dsm5 link f_rw bold">site</a>
      <br/>Open the Terminal app
      <br/>git clone https://github.com/tcew/OCCA2
      <br/>This last command makes a directory OCCA2 and pulls the OCCA source into the new OCCA2 directory
      <br/>Copy the path to the OCCA directory, you can see this by typing
      <br/>pwd
      <br/>on the terminal
    </div>

    <h4 class="ui dividing header"> Installation </h4>
    <div class="dsm5 indent1">
      Install Xcode from the App Store
      <br/>Install the developer tools in XCode (Add SS)
      <br/>Go to the OCCA2 directory
      <br/>We're going to save the directory in a variable OCCA_DIR by using
      <br/>export OCCA_DIR=/absolute/path/to/OCCA2 # For instance: export OCCA_DIR=/Users/dsm5/gitRepos/OCCA2
      <br/>You'll need this when using OCCA, so it's handy to type the line above in your shell init script (like ~/.bashrc or ~/.profile)
      <br/>Now we'll compile with
      <br/>make -j 8 # Compile in parallel with 8 threads, feel free to reduce the number
      <br/>Although the OCCA API is ported to multiple languages, the back-end is mainly in C++.
           To specify the C++ compiler for compiling OCCA, you can use
      <br/>make -j 8 CXX="clang++" # Where g++ or icpc, for example, could be used instead of clang++
      <br/>Check [Specialized Options] for more options
    </div>

    <h4 class="ui dividing header"> Running addVectors </h4>
    <div class="dsm5 indent1">
    </div>
  </div>

  <h2 class="ui dividing header"> Windows </h2>
  <div class="dsm5 indent1">
    <h4 class="ui dividing header"> Downloading Source </h4>
    <div class="dsm5 indent1">
      ?
    </div>

    <h4 class="ui dividing header"> Installation </h4>
    <div class="dsm5 indent1">
      ?
    </div>

    <h4 class="ui dividing header"> Running addVectors </h4>
    <div class="dsm5 indent1">
      ?
    </div>
  </div>

  <h2 class="ui dividing header"> Specialized Options </h2>
  <div class="dsm5 indent1">
    <h4 class="ui dividing header"> Compiling Options </h4>
    <div class="dsm5 indent1">
    </div>

    <h4 class="ui dividing header"> Run-time Options </h4>
    <div class="dsm5 indent1">
    </div>
  </div>

</div> <!--[ id_body ]-->

<?php include($_SERVER['DOCUMENT_ROOT'] . '/footer.php'); ?>
