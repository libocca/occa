<?php
function addHeader($pageTitle){
  ob_start();

  include($_SERVER['DOCUMENT_ROOT'] . '/header.php');
  $headerContents = ob_get_contents();

  ob_end_clean();

  echo str_replace('<!--TITLE-->', $pageTitle, $headerContents);
}

function absInclude($path){
  include($_SERVER['DOCUMENT_ROOT'] . $path);
}

function addSelected($tag){
  if(basename($_SERVER['PHP_SELF'], '.php') == $tag)
    echo 'selected';
  else if(dirname($_SERVER['PHP_SELF']) == '/' . $tag)
    echo 'selected';
}

function highlight($content){
  echo '<span class="f_rw bold highlight">' . $content . '</span>';
}

function addCodeFromFile($filename, $language = 'c_cpp'){
  echo '<pre class="code block" language="' . $language . '">' . file_get_contents($_SERVER['DOCUMENT_ROOT'] . $filename) . '</pre>';
}

function addCopyright($startYear){
  if(intval($startYear) == @date('Y'))
    echo '&copy ' . intval($startYear);
  else
    echo '&copy ' . intval($startYear) . ' - ' . @date('Y');
}
?>
