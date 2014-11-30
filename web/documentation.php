<?php include($_SERVER['DOCUMENT_ROOT'] . '/main.php'); ?>
<?php addHeader('OCCA: Documentation') ?>

<?php absInclude("/menu.php"); ?>

<?php absInclude("/sidebarStart.php"); ?>

<div class="entry1"><a href="#Introduction">1. Introduction</a></div>

<div class="entry1"><a href="#Host-API">2. Host-API    </a></div>
<div class="entry2"><a href="/documentation/hostAPI/C.php"      >2.2 C       </a></div>
<div class="entry2"><a href="/documentation/hostAPI/CPP.php"    >2.1 C++     </a></div>
<div class="entry2"><a href="/documentation/hostAPI/CS.php"     >2.3 C#      </a></div>
<div class="entry2"><a href="/documentation/hostAPI/Fortran.php">2.5 Fortran </a></div>
<div class="entry2"><a href="/documentation/hostAPI/Julia.php"  >2.6 Julia   </a></div>
<div class="entry2"><a href="/documentation/hostAPI/Python.php" >2.4 Python  </a></div>
<div class="entry2"><a href="/documentation/hostAPI/Matlab.php" >2.7 Matlab  </a></div>

<div class="entry1"><a href="#Device-API">3. Device-API  </a></div>
<div class="entry2"><a href="/documentation/deviceAPI/OKL.php">3.1 OKL </a></div>
<div class="entry2"><a href="/documentation/deviceAPI/OFL.php">3.2 OFL </a></div>

<?php absInclude("/sidebarEnd.php"); ?>

<div id="id_body" class="documentation fixed body">

  <h2 id="Introduction" class="ui dividing header"> Quick Introduction </h2>

  <h2 id="Host-API" class="ui dividing header"> Host API </h2>

  <div class="api buttons wrapper">
    <a href="/documentation/hostAPI/C.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img src="/images/apiLogos/cLogo.png" style="height:90px; width:90px; margin: 5px;"/>
        </div>
        <div class="name"> C </div>
      </div>
    </a>
    <a href="/documentation/hostAPI/CPP.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img src="/images/apiLogos/cppLogo.png" style="height:90px; width:90px; margin: 5px;"/>
        </div>
        <div class="name"> C++ </div>
      </div>
    </a>
    <a href="/documentation/hostAPI/CS.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img src="/images/apiLogos/csLogo.png" style="height:90px; width:90px; margin: 5px;"/>
        </div>
        <div class="name"> C# </div>
      </div>
    </a>
    <a href="/documentation/hostAPI/Fortran.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img src="/images/apiLogos/f90Logo.png" style="height:90px; width:90px; margin: 5px;"/>
        </div>
        <div class="name"> Fortran </div>
      </div>
    </a>
    <a href="/documentation/hostAPI/Julia.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img src="/images/apiLogos/julia.png"/>
        </div>
        <div class="name"> Julia </div>
      </div>
    </a>
    <a href="/documentation/hostAPI/Python.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img src="/images/apiLogos/python.png"/>
        </div>
        <div class="name"> Python </div>
      </div>
    </a>
    <a href="/documentation/hostAPI/Matlab.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img src="/images/apiLogos/matlab.png"/>
        </div>
        <div class="name"> MATLAB </div>
      </div>
    </a>
  </div>

  <h2 id="Device-API" class="ui dividing header"> Device API </h2>

  <div class="api buttons wrapper">
    <a href="/documentation/deviceAPI/OKL.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img src="/images/general/mu.png"/>
        </div>
        <div class="name"> OKL </div>
      </div>
    </a>
    <a href="/documentation/deviceAPI/OFL.php" class="link">
      <div class="ui segment">
        <div class="wrapper">
          <img src="/images/general/mu.png"/>
        </div>
        <div class="name"> OFL </div>
      </div>
    </a>
  </div>

</div> <!--[ id_body ]-->

<?php include($_SERVER['DOCUMENT_ROOT'] . '/footer.php'); ?>
