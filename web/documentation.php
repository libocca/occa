<?php include($_SERVER['DOCUMENT_ROOT'] . '/main.php'); ?>
<?php addSidebarHeader('OCCA: Documentation') ?>

<?php startSidebar(); ?>

<div class="entry1"><a href="#API">1. API Reference </a></div>
<div class="entry2"><a href="/documentation/API/C.php"      >1.2 C       </a></div>
<div class="entry2"><a href="/documentation/API/CPP.php"    >1.1 C++     </a></div>
<div class="entry2"><a href="/documentation/API/CS.php"     >1.3 C#      </a></div>
<div class="entry2"><a href="/documentation/API/Fortran.php">1.5 Fortran </a></div>
<div class="entry2"><a href="/documentation/API/Julia.php"  >1.6 Julia   </a></div>
<div class="entry2"><a href="/documentation/API/Python.php" >1.4 Python  </a></div>
<div class="entry2"><a href="/documentation/API/Matlab.php" >1.7 Matlab  </a></div>

<div class="entry1"><a href="#Kernel-Languages">2. Kernel Languages </a></div>
<div class="entry2"><a href="/documentation/kernelLanguages/OKL.php">2.1 OKL </a></div>
<div class="entry2"><a href="/documentation/kernelLanguages/OFL.php">2.2 OFL </a></div>

<?php endSidebar(); ?>

<?php absInclude("/menu.php"); ?>

<div id="id_body" class="documentation fixed body">

  <h2 id="API" class="ui dividing header"> API Reference </h2>

  <div class="api buttons wrapper">
    <a href="/documentation/API/C.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img class="apiLogo" src="/images/apiLogos/cLogo.png"></img>
        </div>
        <div class="name"> C </div>
      </div>
    </a>
    <a href="/documentation/API/CPP.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img class="apiLogo" src="/images/apiLogos/cppLogo.png"></img>
        </div>
        <div class="name"> C++ </div>
      </div>
    </a>
    <a href="/documentation/API/CS.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img class="apiLogo" src="/images/apiLogos/csLogo.png"></img>
        </div>
        <div class="name"> C# </div>
      </div>
    </a>
    <a href="/documentation/API/Fortran.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img class="apiLogo" src="/images/apiLogos/f90Logo.png"></img>
        </div>
        <div class="name"> Fortran </div>
      </div>
    </a>
    <a href="/documentation/API/Julia.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img class="apiLogo" src="/images/apiLogos/julia.png"></img>
        </div>
        <div class="name"> Julia </div>
      </div>
    </a>
    <a href="/documentation/API/Python.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img class="apiLogo" src="/images/apiLogos/python.png"></img>
        </div>
        <div class="name"> Python </div>
      </div>
    </a>
    <a href="/documentation/API/Matlab.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img class="apiLogo" src="/images/apiLogos/matlab.png"></img>
        </div>
        <div class="name"> MATLAB </div>
      </div>
    </a>
  </div>

  <h2 id="Kernel-Languages" class="ui dividing header"> Kernel Languages </h2>

  <div class="api buttons wrapper">
    <a href="/documentation/kernelLanguages/OKL.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img class="apiLogo" src="/images/apiLogos/oklLogo.png"></img>
        </div>
        <div class="name"> OKL </div>
      </div>
    </a>
    <a href="/documentation/kernelLanguages/OFL.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img class="apiLogo" src="/images/apiLogos/oflLogo.png"></img>
        </div>
        <div class="name"> OFL </div>
      </div>
    </a>
  </div>

</div> <!--[ id_body ]-->

<?php addFooter() ?>
