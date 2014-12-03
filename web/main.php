<?php
function addHeader($pageTitle){
  ob_start();

  include($_SERVER['DOCUMENT_ROOT'] . '/header.php');
  $headerContents = ob_get_contents();

  ob_end_clean();

  echo str_replace('<!--TITLE-->', $pageTitle, $headerContents);
}

function addFooter(){
  include($_SERVER['DOCUMENT_ROOT'] . '/footer.php');
}

function absInclude($path){
  include($_SERVER['DOCUMENT_ROOT'] . $path);
}

function addSelected($tag){
  if(basename($_SERVER['PHP_SELF'], '.php') == $tag){
    echo 'selected';
  }
  else {
    $path     = dirname($_SERVER['PHP_SELF'] . '.php');
    $pathDirs = explode('/', $path);

    if($pathDirs[1] == $tag)
      echo 'selected';
  }
}

function highlight($content){
  echo '<span class="highlight">' . $content . '</span>';
}

function addCodeFromFile($filename, $language = 'c_cpp'){
  echo '<pre class="code block" language="' . $language . '">' . file_get_contents($_SERVER['DOCUMENT_ROOT'] . $filename) . '</pre>';
}

function startFunctionAPI($title){
  echo
    '<div class="ui fluid api accordion">';

  addFunctionTitleAPI($title);
}

function nextFunctionAPI($title){
  echo
    '  </div>';

  addFunctionTitleAPI($title);
}

function addFunctionTitleAPI($title){
  echo
    '  <div class="title"><i class="dropdown icon"></i>' .
    $title .
    '  </div>' .
    '  <div class="content">';
}

function endFunctionAPI(){
  echo
    '  </div>' .
    '</div>'   .
    '<div class="dSpacing2"></div>';
}

function addCopyright($startYear){
  if(intval($startYear) == @date('Y'))
    echo '&copy ' . intval($startYear);
  else
    echo '&copy ' . intval($startYear) . ' - ' . @date('Y');
}
?>
