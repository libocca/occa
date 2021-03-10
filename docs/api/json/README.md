<h1 id="occa::json">
 <a href="#/api/json/" class="anchor">
   <span>occa::json</span>
  </a>
</h1>

<h2 id="description">
 <a href="#/api/json/?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>


A [occa::json](/api/json/) object stores data in the same way specified by the JSON standard.
It's used across the OCCA library as a way to flexibly pass user configurations.

<h2 id="types">
 <a href="#/api/json/?id=types" class="anchor">
   <span>Types</span>
  </a>
</h2>

There are 6 basic types a json object can be at a time:
- String
- Number
- Boolean
- NULL
- Array of json objects
- Map of string keys to json objects

<h2 id="type checking">
 <a href="#/api/json/?id=type checking" class="anchor">
   <span>Type checking</span>
  </a>
</h2>

There is a method provided check for each type

- [isString](/api/json/isString)
- [isNumber](/api/json/isNumber)
- [isBool](/api/json/isBool)
- [isNull](/api/json/isNull)
- [isObject](/api/json/isObject)
- [isArray](/api/json/isArray)

<h2 id="type casting">
 <a href="#/api/json/?id=type casting" class="anchor">
   <span>Type casting</span>
  </a>
</h2>

 There is also a method to enforce the json object to be a specific type

- [asString](/api/json/asString)
- [asNumber](/api/json/asNumber)
- [asBoolean](/api/json/asBoolean)
- [asNull](/api/json/asNull)
- [asObject](/api/json/asObject)
- [asArray](/api/json/asArray)

<h2 id="data access">
 <a href="#/api/json/?id=data access" class="anchor">
   <span>Data access</span>
  </a>
</h2>

Accessing and setting data can be done through the [operator []](/api/json/operator_brackets).
To make it simpler to access nested structures, we support passing `/`-delimited paths

For example, if we wanted to build

```js
{
  "a": {
    "b": {
      "c": "hello world"
    }
  }
}
```

we could do it two ways:

```cpp
occa::json j;
j["a"]["b"]["c"] = "hello world';
```

or a the more compact way:

```cpp
occa::json j;
j["a/b/c"] = "hello world';
```

If for some reason there needs to be a `/` in the key name, use the [set](/api/json/set) method instead

For example, building

```js
{
  "a/b/c": "hello world"
}
```

would be done through

```cpp
occa::json j;
j.set("a/b/c", "hello world');
```

<h2 id="decoding">
 <a href="#/api/json/?id=decoding" class="anchor">
   <span>Decoding</span>
  </a>
</h2>

- [parse](/api/json/parse) can be used to parse a string to a json object.
- [read](/api/json/read) is the same as [parse](/api/json/parse) but reads and parses a file instead.

<h2 id="encoding">
 <a href="#/api/json/?id=encoding" class="anchor">
   <span>Encoding</span>
  </a>
</h2>
- [dump](/api/json/dump) produces the JSON string associated with the stored data.
- [write](/api/json/write) is the same as [dump](/api/json/dump) but saves the output in a file.