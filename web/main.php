<?php

function _addHeader($pageTitle){
  ob_start();

  include($_SERVER['DOCUMENT_ROOT'] . '/header.php');
  $headerContents = ob_get_contents();

  ob_end_clean();

  echo str_replace('<!--TITLE-->', $pageTitle, $headerContents);
}

function addHeader($pageTitle){
  _addHeader($pageTitle);
  startBody();
  startBodyWrapper();
}

function addSidebarHeader($pageTitle){
  _addHeader($pageTitle);
}

function startBody(){
  echo ('<body class="pushable">');
}

function startBodyWrapper(){
  echo ('<div class="pusher" id="id_bodyWrapper">' .
        '  <div id="id_bodyWrapper2">');
}

function endBody(){
  echo ('</body>');
}

function endBodyWrapper(){
  echo ('  </div>' .
        '</div>');
}

function startSidebar($sidebarHeight = 0, $sidebarWidth = 0){
  startBody();

  $style = '';
  $bStyle = '';

  if(0 < $sidebarHeight){
    $style = ('style="height: ' .
              strval($sidebarHeight) .
              'px !important;');

    if(0 < $sidebarWidth){
      $style .= ('width: ' .
                 strval($sidebarWidth) .
                 'px !important;');

      $bStyle = 'style="left: ' . strval($sidebarWidth - 2) . 'px"';
    }

    $style .= '"';
  }

  echo ('<div class="ui left overlay sidebar body" ' . $style . '>' .
        '  <div class="content title"><a href="#">Content</a></div>' .
        '    <div class="ui black huge right sidebarItem button" ' .$bStyle . '>' .
        '      <i class="icon list layout"></i>' .
        '      <span class="text f_os light">Contents</span>' .
        '    </div>' .
        '    <div class="wrapper">');
}

function endSideBar(){
  echo ('    </div>' .
        '  </div>' .
        '</div>');

  startBodyWrapper();
}

function addFooter(){
  endBodyWrapper();
  endBody();

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

function warning($content){
  echo '<span class="warning">' . $content . '</span>';
}

function ready($color){
  switch($color){
  case 'g': echo '<i class="c ready g check square icon"></i>'; break;
  case 'y': echo '<i class="c ready y check square icon"></i>'; break;
  case 'r': echo '<i class="c ready r check square icon"></i>'; break;
  }
}

function emptyReady(){
  echo '<div style="width: 20px; height: 1px; display: inline-block"></div>';
}

function addCodeFromFile($filename, $language = 'c_cpp'){
  echo '<pre class="code block" language="' . $language . '">' . file_get_contents($_SERVER['DOCUMENT_ROOT'] . $filename) . '</pre>';
}

function startFunctionAPI($title, $isDone = 'done'){
  echo
    '<div class="ui fluid styled api accordion">';

  addFunctionTitleAPI($title, $isDone);
}

function nextFunctionAPI($title, $isDone = 'done'){
  echo
    '  </div>';

  addFunctionTitleAPI($title, $isDone);
}

function addFunctionTitleAPI($title, $isDone = 'done'){
  echo
    '  <div class="api title ' . $isDone . '">' .
    '    <pre class="cpp api code block">' . $title . '</pre>' .
    '  </div>' .
    '  <div class="api content">';
  /* echo */
  /*   '  <div class="api title active ' . $isDone . '">' . */
  /*   '    <pre class="cpp api code block">' . $title . '</pre>' . */
  /*   '  </div>' . */
  /*   '  <div class="api content active">'; */
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
