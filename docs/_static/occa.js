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
occa.codeToMarkdown = (language, code) => {
  const styledCode = Prism.highlight(code,
                                     Prism.languages[language],
                                     language);
  return (
    `        <pre data-lang="${language}">`
      + `          <code class="lang-${language}">`
      + `${styledCode}\n`
      + '          </code>'
      + '        </pre>'
  );
}

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
  `      <md-tab id="${tab}" md-label="${tab}">\n`
    + occa.tokensToMarkdown(content)
    + '      </md-tab>'
);

occa.getTabs = (namespace, tabs) => {
  const content = tabs.map(occa.getTab).join('\n');

  const tab     = `vm.$data.tabNamespaces['${namespace}']`;
  const onClick = `(tab) => vm.onTabChange('${namespace}', tab)`;

  return (
    '<template>\n'
      + '  <div>\n'
      + '    <md-tabs\n'
      + `      v-bind:md-active-tab="${tab}"\n`
      + `      @md-changed="${onClick}"\n`
      + '    >\n'
      + `${content}\n`
      + '    </md-tabs>\n'
      + '  </div>\n'
      + '</template>\n'
  );
};

occa.parseTabs = (namespace, content) => {
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

  if (!newParts.length) {
    return [];
  }

  if (!(namespace in vm.$data.tabNamespaces)) {
    Vue.set(vm.tabNamespaces, namespace, newParts[0].tab);
  }

  return occa.getTabs(namespace, newParts);
};

occa.addTabs = (content) => {
  const re = /\n::: tabs (.*)\n([\s\S]*?)\n:::\n/g;
  const parts = [];
  var lastIndex = 0;
  while ((match = re.exec(content)) != null) {
    const [fullMatch, namespace, tabContent] = match;

    parts.push(content.substring(lastIndex, match.index));
    parts.push(occa.parseTabs(namespace, tabContent));

    lastIndex = match.index + fullMatch.length;
  }
  parts.push(content.substring(lastIndex));

  return parts.join('\n');
};
//======================================

occa.docsifyPlugin = (hook, vm) => {
  vm.$data = { foo: 'hi' };
  hook.beforeEach((content) => {
    content = occa.addHeader(vm, content);
    content = occa.addTabs(content);
    content = occa.addFooter(content);
    return content;
  });
};
