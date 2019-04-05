namespace occa {
  namespace lang {
    template <class attributeType>
    void parser_t::addAttribute() {
      attributeType *attr = new attributeType();
      const std::string name = attr->name();

      OCCA_ERROR("Attribute [" << name << "] already exists",
                 attributeMap.find(name) == attributeMap.end());

      attributeMap[name] = attr;
    }
  }
}
