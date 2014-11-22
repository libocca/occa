<!DOCTYPE html>
<html>
  <head>
    <!-- Site Properities -->
    <title>OCCA</title>

    <link rel="stylesheet" type="text/css" href="/library/css/semantic.min.css">

    <link type="text/css" rel="stylesheet" href="/main.css">

    <script>
      (function () {
      var
      eventSupport = ('querySelector' in document && 'addEventListener' in window)
      jsonSupport  = (typeof JSON !== 'undefined'),
      jQuery       = (eventSupport && jsonSupport)
      ? '/library/js//jquery.min.js'
      : '/library/js//jquery.legacy.min.js'
      ;
      document.write('<script src="' + jQuery + '"><\/script>');
          }());
      </script>

      <script type="text/javascript" src="/library/js/jquery.address.js"></script>
      <script type="text/javascript" src="/library/js/semantic.min.js"></script>

      <script type="text/javascript" src="/main.js"></script>
  </head>

  <body>
    <div id="id_bodyWrapper">
      <div id="id_bodyWrapper2">

        <?php
           $currentTab = "about";
           include("menu.php");
        ?>

        <div id="id_body">

          <!--[BODY GOES HERE]-->

        </div> <!--[ id_body ]-->
      </div> <!--[ id_bodyWrapper2 ]-->
    </div> <!--[ id_bodyWrapper ]-->

    <?php include("footer.php") ?>
  </body>
</html>
