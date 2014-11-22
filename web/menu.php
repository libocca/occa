<div class="ui top menu" id="id_topMenu">
  <div class="top wrapper">
    <a class="item" id="id_logo_div" href="/index.php">
      <img id="id_top_logo"    class="top logo"    src="/images/blueOccaLogo.png"/>
      <img id="id_bottom_logo" class="bottom logo" src="/images/blackOccaLogo.png"/>
    </a>

    <div class="right menu">
      <a class="light topMenu black item <?php if($currentTab == 'tutorial'){echo 'active';}?>" id="id_tutorial_item" href="/tutorial.php">
        Tutorial
      </a>
      <a class="light topMenu black item <?php if($currentTab == 'downloads'){echo 'active';}?>" id="id_downloads_item" href="/downloads.php">
        Downloads
      </a>
	    <a class="light topMenu black item <?php if($currentTab == 'documentation'){echo 'active';}?>" id="id_documentation_item" href="/documentation.php">
	      Documentation
	    </a>
	    <a class="light topMenu black item <?php if($currentTab == 'about'){echo 'active';}?>" id="id_about_item" href="/about.php">
	      About
	    </a>
      <a class="light topMenu black item <?php if($currentTab == 'support'){echo 'active';}?>" id="id_support_item" href="/support.php">
        Support
      </a>
    </div> <!--[ right menu ]-->
  </div> <!--[ top wrapper ]-->
</div> <!--[ id_topMenu ]-->
