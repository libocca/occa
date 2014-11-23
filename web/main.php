<?php
function absInclude($path){
  include($_SERVER['DOCUMENT_ROOT'] . $path);
}

function addSelected($tag){
  if(basename($_SERVER['PHP_SELF'], '.php') == $tag)
    echo "selected";
  else if(dirname($_SERVER['PHP_SELF']) == '/' . $tag)
    echo "selected";
}

function addCopyright($startYear){
  if(intval($startYear) == @date('Y'))
    echo '&copy ' . intval($startYear);
  else
    echo '&copy ' . intval($startYear) . ' - ' . @date('Y');
}
?>
