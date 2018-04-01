var occa = occa || {};

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

occa.getOSTab = ({ content, os }) => (
  `      <md-tab id="${os}" md-label="${occa.label[os]}">\n`
    + occa.tokensToMarkdown(content)
    + '      </md-tab>'
);

occa.getOSTabs = (tabs) => (
  occa.getTabs(tabs,
               occa.getOSTab,
               'os')
);

occa.getLanguageTab = ({ language, content }) => (
  `      <md-tab id="${language}" md-label="${occa.label[language]}">\n`
    + occa.codeToMarkdown(language, content)
    + '      </md-tab>'
);

occa.getLanguageTabs = (tabs) => (
  occa.getTabs(tabs,
               occa.getLanguageTab,
               'language')
);

occa.getTabs = (tabs, getTab, tabKey) => {
  const content = tabs.map(getTab).join('\n');

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

occa.parseLanguageTabs = (content) => (
  occa.getLanguageTabs(
    marked.lexer(content)
      .map(({ text: content, lang: language }) => (
        { content, language }
      ))
  )
);

occa.parseOSTabs = (content) => {
  const parts = marked.lexer(content);
  const newParts = [];

  // Skip begin/end of list
  for (var i = 1; i < (parts.length - 1); ++i) {
    // Skip loose_item_start;
    ++i;
    const os = parts[i++].text;
    const start = i++;
    while (parts[i].type !== 'list_item_end') {
      ++i;
    }
    newParts.push({
      os,
      content: parts.slice(start, i),
    });
  }

  return occa.getOSTabs(newParts);
};

occa.parseTabs = (style, content) => {
  if (style === 'language-tabs') {
    return occa.parseLanguageTabs(content);
  }
  return occa.parseOSTabs(content);
};

occa.parse = (content) => {
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
