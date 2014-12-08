<?php include($_SERVER['DOCUMENT_ROOT'] . '/main.php'); ?>
<?php addHeader('OCCA') ?>

<?php absInclude("/menu.php"); ?>

<div style="margin: 20px auto 20px auto; width: 600px; position: relative;">
<img src="/images/logo/occaFlow.png" style="display: block; width: 600px;"></img>

<a href="/documentation/API/C.php"
   style="width: 51px; height: 51px; border-radius: 25px; background-color: transparent; position: absolute; left: 25px; top: 71px;"></a>

<a href="/documentation/API/CPP.php"
   style="width: 51px; height: 51px; border-radius: 25px; background-color: transparent; position: absolute; left: 25px; top: 134px;"></a>

<a href="/documentation/API/CS.php"
   style="width: 51px; height: 51px; border-radius: 25px; background-color: transparent; position: absolute; left: 25px; top: 196px;"></a>

<a href="/documentation/API/Fortran.php"
   style="width: 52px; height: 51px; background-color: transparent; position: absolute; left: 25px; top: 258px;"></a>

<a href="/documentation/API/Python.php"
   style="width: 51px; height: 51px; background-color: transparent; position: absolute; left: 25px; top: 320px;"></a>

<a href="/documentation/API/Julia.php"
   style="width: 53px; height: 51px; background-color: transparent; position: absolute; left: 25px; top: 382px;"></a>

<a href="/documentation/API/Matlab.php"
   style="width: 53px; height: 51px; background-color: transparent; position: absolute; left: 25px; top: 444px;"></a>

<a href="/documentation/kernelLanguages/OKL.php"
   style="width: 51px; height: 51px; border-radius: 25px; background-color: transparent; position: absolute; left: 293px; top: 194px;"></a>

<a href="/documentation/kernelLanguages/OFL.php"
   style="width: 52px; height: 51px; background-color: transparent; position: absolute; left: 293px; top: 320px;"></a>
</div>

<div style="width: 100%; background-color: #EAF5F9; padding: 20px 0 20px 0;">
  <div style="height: 250px; width: 1060px; margin: auto; color: black; margin: 20px auto 20px auto; position: relative; clear: both;">
    <a style="width: 250px; float: left; display: block; position: relative;" href="/downloads.php">
      <div style="margin: auto; width: 80px; color: #6992B5;"><i class="huge download icon"></i></div>
      <p style="text-align: center; color: #6C6C6C; font-size: 18px; font-family: openSansBold; margin: 15px 0 25px 0;">
        Downloads
      </p>
      <div style="margin: 10px; color: #6C6C6C;">
        Source code for OCCA is open-source and available in Github. Additional tools for syntax highlighting are also available.
      </div>
    </a>

    <div style="height: 1px; width: 20px; float: left; display: block; vertical-align: middle; position: relative"></div>

    <a style="width: 250px; float: left; display: block; position: relative" href="/getStarted.php">
      <div style="margin: auto; width: 80px; color: #6992B5;"><i class="huge map marker icon"></i></div>
      <p style="text-align: center; color: #6C6C6C; font-size: 18px; font-family: openSansBold; margin: 15px 0 25px 0">
        Getting Started
      </p>
      <div style="margin: 10px; color: #6C6C6C;">
        Get started by checking the installation guides and provided examples for Linux, Mac OS X and Windows.
      </div>
    </a>

    <div style="height: 1px; width: 20px; float: left; display: block; vertical-align: middle; position: relative"></div>

    <a style="width: 250px; float: left; display: block;" href="/tutorials.php">
      <div style="margin: auto; width: 80px; color: #6992B5;"><i class="huge map icon"></i></div>
      <p style="text-align: center; color: #6C6C6C; font-size: 18px; font-family: openSansBold; margin: 15px 0 25px 0">
        Tutorials
      </p>
      <div style="margin: 10px; color: #6C6C6C;">
        Learn how OCCA works through easy-to-hard sample codes and their explanations.
        We'll go through some good-practice programming for the offloading model.
      </div>
    </a>

    <div style="height: 1px; width: 20px; float: left; display: block; vertical-align: middle; position: relative"></div>

    <a style="width: 250px; float: left; display: block;" href="/documentation.php">
      <div style="margin: auto; width: 80px; color: #6992B5;"><i class="huge settings icon"></i></div>
      <p style="text-align: center; color: #6C6C6C; font-size: 18px; font-family: openSansBold; margin: 15px 0 25px 0">
        Documentation
      </p>
      <div style="margin: 10px; color: #6C6C6C;">
        Check the documentation for our API and kernel languages OKL and OFL.
      </div>
    </a>
  </div>
</div>

<!-- <div style="height: 200px; width: 100%; background-color: black; position: relative; clear: both"> -->
<!--   <div style="height: 200px; width: 1060px; margin: auto; position: relative; overflow: hidden;"> -->
<!--     <div style="height: 180px; background-color: blue; color: black; margin: 10px 10px 10px 10px;"> -->
<!--       Timeline -->
<!--     </div> -->
<!--   </div> -->
<!-- </div> -->

<?php addFooter(); ?>
