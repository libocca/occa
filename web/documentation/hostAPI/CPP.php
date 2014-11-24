<?php include($_SERVER['DOCUMENT_ROOT'] . '/main.php'); ?>
<?php addHeader('OCCA: C++ API') ?>

<?php absInclude("/menu.php"); ?>
<?php absInclude("/documentation/hostAPI/menu.php") ?>

<div id="id_body" class="fixed body">

  <?php addCodeFromFile("/src/addVectors.okl", "c_cpp") ?>

</div> <!--[ id_body ]-->

<?php include($_SERVER['DOCUMENT_ROOT'] . '/footer.php'); ?>
