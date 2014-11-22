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
           $currentTab = "tutorial";
           include("menu.php");
         ?>

        <div id="id_body" class="tutorial">

          <div id="id_tutorialMenu">

          </div> <!--[ id_tutorialMenu ]-->

          <div id="id_tutorialContents">

            <pre id="editor"><?php echo file_get_contents($_SERVER['DOCUMENT_ROOT'] . "/src/addVectors.okl"); ?></pre>

          </div> <!--[ id_tutorialContents ]-->

        </div> <!--[ id_body ]-->
      </div> <!--[ id_bodyWrapper2 ]-->
    </div> <!--[ id_bodyWrapper ]-->

    <?php include("footer.php") ?>

    <script src="/library/js/aceMin/ace.js" type="text/javascript" charset="utf-8"></script>
    <script>
      var editor = ace.edit("editor");
      editor.setTheme("ace/theme/chrome");
      editor.getSession().setMode("ace/mode/c_cpp");
      editor.setReadOnly(true);
      editor.renderer.setShowGutter(false);
      editor.setHighlightActiveLine(false);
      editor.setDisplayIndentGuides(false);
      editor.setShowPrintMargin(false);
      editor.setOption("maxLines", 20);
    </script>
  </body>
</html>
