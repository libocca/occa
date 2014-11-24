<?php include($_SERVER['DOCUMENT_ROOT'] . '/main.php'); ?>
<?php addHeader('OCCA: Get Started') ?>

<?php absInclude("/menu.php"); ?>

<?php absInclude("/sidebarStart.php"); ?>

<div class="entry1"><a href="#Linux"  >1. Linux</a></div>
<div class="entry2"><a href="#Linux-Download"     >1.1 Downloading Source</a></div>
<div class="entry2"><a href="#Linux-Installation" >1.2 Installation</a></div>
<div class="entry2"><a href="#Linux-Example"      >1.2 Running addVectors</a></div>

<div class="entry1"><a href="#MacOSX" >2. Mac OS X</a></div>
<div class="entry2"><a href="#MacOSX-Download"     >2.1 Downloading Source</a></div>
<div class="entry2"><a href="#MacOSX-Installation" >2.2 Installation</a></div>
<div class="entry2"><a href="#MacOSX-Example"      >2.2 Running addVectors</a></div>

<div class="entry1"><a href="#Windows">3. Windows</a></div>
<div class="entry2"><a href="#Windows-Download"     >3.1 Downloading Source</a></div>
<div class="entry2"><a href="#Windows-Installation" >3.2 Installation</a></div>
<div class="entry2"><a href="#Windows-Example"      >3.2 Running addVectors</a></div>

<div class="entry1"><a href="#Specialized-Options">4. Specialized Options</a></div>

<?php absInclude("/sidebarEnd.php"); ?>

<div id="id_body" class="fixed body">

  <h2 id="Linux" class="ui dividing header"> Linux </h2>
  <div class="dsm5 indent1">
    <h4 class="ui dividing header"> Downloading Source </h4>
    <div class="dsm5 indent1">
      If you don't have git installed, install it with

      <pre class="code block udSpacing1" language="sh">
sudo apt-get install git
git clone https://github.com/tcew/OCCA2</pre>

      This last command makes a directory OCCA2 and pulls the OCCA source into the new OCCA2 directory.
      Copy the path to the OCCA directory and you can check your absolute path by typing

      <pre class="code block udSpacing1" language="sh">pwd</pre>

      on the terminal
    </div>

    <h4 class="ui dividing header"> Installation </h4>
    <div class="dsm5 indent1">
      Install make

      <pre class="code block udSpacing1" language="sh">
sudo apt-get install make</pre>

      Go to the OCCA2 directory.
      We're going to save the directory in a variable OCCA_DIR by using

      <pre class="code block udSpacing1" language="sh">
export OCCA_DIR=/absolute/path/to/OCCA2 # For instance: export OCCA_DIR=/home/dsm5/gitRepos/OCCA2</pre>

      You'll need this when using OCCA, so it's handy to type the line above in your shell init script
      (like <?php highlight('~/.bashrc') ?> or <?php highlight('~/.profile') ?>).

      Now we'll compile with

      <pre class="code block udSpacing1" language="sh">
make -j 8 # Compile in parallel with 8 threads, feel free to reduce the number</pre>

      Although the OCCA API is ported to multiple languages, the back-end is mainly in C++.
           To specify the C++ compiler for compiling OCCA, you can use

      <pre class="code block udSpacing1" language="sh">
make -j 8 CXX="clang++" # Where g++ or icpc, for example, could be used instead of clang++</pre>

      Check
      <a href="" class="dsm5 link f_rw bold">Specialized Options</a>
      for more options
    </div>

    <h4 class="ui dividing header"> Running addVectors </h4>
    <div class="dsm5 indent1">
    </div>
  </div>

  <h2 id="MacOSX" class="ui dividing header"> Mac OS X </h2>
  <div class="dsm5 indent1">
    <h4 class="ui dividing header"> Downloading Source </h4>
    <div class="dsm5 indent1">
      If you don't have git installed, you can download Git from their
      <a href="http://git-scm.com/download/mac" class="dsm5 link f_rw bold">site</a>
      <br/>Open the Terminal app to get the OCCA source code

      <pre class="code block udSpacing1" language="sh">
git clone https://github.com/tcew/OCCA2</pre>

      This last command makes a directory OCCA2 and pulls the OCCA source into the new OCCA2 directory.
      Copy the path to the OCCA directory and you can check your absolute path by typing

      <pre class="code block udSpacing1" language="sh">pwd</pre>

      on the terminal
    </div>

    <h4 class="ui dividing header"> Installation </h4>
    <div class="dsm5 indent1">
      You'll need some compiler tools which requires the developer tools in Xcode from the App Store

      <br/>Go to the OCCA2 directory.
      We're going to save the directory in a variable OCCA_DIR by using

      <pre class="code block udSpacing1" language="sh">
export OCCA_DIR=/absolute/path/to/OCCA2 # For instance: export OCCA_DIR=/home/dsm5/gitRepos/OCCA2</pre>

      You'll need this when using OCCA, so it's handy to type the line above in your shell init script
      (like <?php highlight('~/.bashrc') ?> or <?php highlight('~/.profile') ?>).

      Now we'll compile with

      <pre class="code block udSpacing1" language="sh">
make -j 8 # Compile in parallel with 8 threads, feel free to reduce the number</pre>

      Although the OCCA API is ported to multiple languages, the back-end is mainly in C++.
           To specify the C++ compiler for compiling OCCA, you can use

      <pre class="code block udSpacing1" language="sh">
make -j 8 CXX="clang++" # Where g++ or icpc, for example, could be used instead of clang++</pre>

      Check
      <a href="" class="dsm5 link f_rw bold">Specialized Options</a>
      for more options
    </div>

    <h4 class="ui dividing header"> Running addVectors </h4>
    <div class="dsm5 indent1">
    </div>
  </div>

  <h2 id="Windows" class="ui dividing header"> Windows </h2>
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
