<div class="ui top menu" id="id_topMenu">
  <div class="top wrapper">
    <a class="item" id="id_logo_div" href="/">
      <img id="id_top_logo"    class="top logo"    src="/images/logo/blueOccaLogo.png"></img>
      <img id="id_bottom_logo" class="bottom logo" src="/images/logo/blackOccaLogo.png"></img>
    </a>

    <div class="right menu">
      <a class="light topMenu black item <?php addSelected('downloads') ?>"
         href="/downloads.php">
        Downloads
      </a>

	    <div class="ui light topMenu black simple dropdown item <?php addSelected('getStarted') ?>">
        <a class="topMenuDropdownItem" href="/getStarted.php">
          Get Started
        </a>
        <div class="menu">
          <a class="item" href="/getStarted.php#Linux"  > Linux    </a>
          <a class="item" href="/getStarted.php#MacOSX" > Mac OS X </a>
          <a class="item" href="/getStarted.php#Windows"> Windows  </a>
        </div>
      </div>

      <a class="light topMenu black item <?php addSelected('tutorials') ?>"
         href="/tutorials.php">
        Tutorials
      </a>

	    <div class="ui light topMenu black simple dropdown item <?php addSelected('documentation') ?>">
        <a class="topMenuDropdownItem" href="/documentation.php">
          Documentation
        </a>
        <div class="menu">
          <div class="ui dropdown item">
            <a class="noDecor f_14" href="/documentation.php#API"> API Reference <i class="dropdown icon"></i></a>
            <div class="menu">

              <a class="item" href="/documentation/API/C.php"      > <?php ready('r'); ?> C       </a>
              <a class="item" href="/documentation/API/CPP.php"    > <?php ready('y'); ?> C++     </a>
              <a class="item" href="/documentation/API/CS.php"     > <?php ready('r'); ?> C#      </a>
              <a class="item" href="/documentation/API/Fortran.php"> <?php ready('r'); ?> Fortran </a>
              <a class="item" href="/documentation/API/Python.php" > <?php ready('r'); ?> Python  </a>
              <a class="item" href="/documentation/API/Julia.php"  > <?php ready('r'); ?> Julia   </a>
              <a class="item" href="/documentation/API/Matlab.php" > <?php ready('r'); ?> MATLAB  </a>
            </div>
          </div>
          <div class="ui dropdown item">
            <a class="noDecor f_14" href="/documentation.php#kernelLanguages"> Kernel Languages <i class="dropdown icon"></i></a>
            <div class="menu">
              <a class="item" href="/documentation/kernelLanguages/OKL.php"> <?php ready('r'); ?> OKL </a>
              <a class="item" href="/documentation/kernelLanguages/OFL.php"> <?php ready('r'); ?> OFL </a>
            </div>
          </div>
        </div>
      </div>

	    <a class="light topMenu black item <?php addSelected('aboutUs') ?>"
         href="/aboutUs.php">
        About Us
      </a>
    </div> <!--[ right menu ]-->
  </div> <!--[ top wrapper ]-->
</div> <!--[ id_topMenu ]-->
