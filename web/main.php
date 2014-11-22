<?php
function absInclude($path){
  include($_SERVER['DOCUMENT_ROOT'] . $path);
}

function addActive($tag){
  if(basename($_SERVER['PHP_SELF'], '.php') == $tag)
    echo "active";
  else if(dirname($_SERVER['PHP_SELF']) == '/' . $tag)
    echo "active";
}
?>