var occa = occa || {};

//---[ Header & Footer ]----------------
// Credit to QingWei-Li/docsify for the template
occa.addHeader = (vm, content) => {
  const url = `https://github.com/libocca/occa/blob/master/docs/${vm.route.file}`;
  return (
    '<div\n'
      + '  style="position: absolute; top: 0"'
      + '>\n'
      + `  [Edit Source](${url})\n`
      + '</div>\n'
      + content
  );
};

occa.addFooter = (content) => (
  content
    + '\n'
    + '---\n'
    + '<span\n'
    + '  style="color: #B2B3BA; position: absolute; bottom: 2em;"\n'
    + '>\n'
    + `  © Copyright 2014 - ${(new Date()).getFullYear()}, David Medina and Tim Warburton.\n`
    + '</span>\n'
);
//======================================

//---[ Tabs ]---------------------------
occa.label = {
  cpp: 'C++',
  c: 'C',
  python: 'Python',

  linux: 'Linux',
  macos: 'MacOS',
  windows: 'Windows',
};

occa.onChange = {
  language: 'onLanguageChange',
  os: 'onOSChange',
}

occa.codeToMarkdown = (language, content) => (
  `        <pre data-lang="${language}">`
    + `          <code class="lang-${language}">`
    + `${content}\n`
    + '          </code>'
    + '        </pre>'
);

occa.tokenToMarkdown = (token) => {
  switch (token.type) {
  case 'code':
    return occa.codeToMarkdown(token.lang, token.text);
  default:
    return '';
  }
};

occa.tokensToMarkdown = (tokens) => (
  tokens.map(occa.tokenToMarkdown).join('\n')
);

occa.getTab = ({ tab, content }) => (
  `      <md-tab id="${tab}" md-label="${occa.label[tab]}">\n`
    + occa.tokensToMarkdown(content)
    + '      </md-tab>'
);

occa.getTabs = (tabKey, tabs) => {
  const content = tabs.map(occa.getTab).join('\n');

  return (
    '<template>\n'
      + '  <div>\n'
      + '    <md-tabs\n'
      + `      v-bind:md-active-tab="vm.$data.${tabKey}"\n`
      + `      @md-changed="vm.${occa.onChange[tabKey]}"\n`
      + '    >\n'
      + `${content}\n`
      + '    </md-tabs>\n'
      + '  </div>\n'
      + '</template>\n'
  );
};

occa.parseTabs = (style, content) => {
  const parts = marked.lexer(content);
  const newParts = [];

  // Skip begin/end of list
  for (var i = 1; i < (parts.length - 1); ++i) {
    // Skip loose_item_start;
    ++i;
    const tab = parts[i++].text;
    const start = i++;
    while (parts[i].type !== 'list_item_end') {
      ++i;
    }
    newParts.push({
      tab,
      content: parts.slice(start, i),
    });
  }

  const tabKey = ((style === 'language-tabs')
                  ? 'language'
                  : 'os');

  return occa.getTabs(tabKey, newParts);
};

occa.addTabs = (content) => {
  const re = /\n::: (language-tabs|os-tabs)\n([\s\S]*?)\n:::\n/g;
  const parts = [];
  var lastIndex = 0;
  while ((match = re.exec(content)) != null) {
    const [fullMatch, tabStyle, tabContent] = match;

    parts.push(content.substring(lastIndex, match.index));
    parts.push(occa.parseTabs(tabStyle, tabContent));

    lastIndex = match.index + fullMatch.length;
  }
  parts.push(content.substring(lastIndex));

  return parts.join('\n');
};
//======================================

occa.docsifyPlugin = (hook, vm) => {
  hook.beforeEach((content) => {
    content = occa.addHeader(vm, content);
    content = occa.addTabs(content);
    content = occa.addFooter(content);
    return content;
  });
};
