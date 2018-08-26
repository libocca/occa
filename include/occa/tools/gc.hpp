/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#ifndef OCCA_TOOLS_GC_HEADER
#define OCCA_TOOLS_GC_HEADER

#include <stdint.h>
#include <cstddef>
#include <map>
#include <vector>

namespace occa {
  namespace gc {
    class withRefs {
    private:
      int refs;

    public:
      withRefs();

      int getRefs() const;
      void addRef();
      int removeRef();

      void setRefs(const int refs_);
      void dontUseRefs();
    };

    class ringEntry_t {
    public:
      ringEntry_t *leftRingEntry;
      ringEntry_t *rightRingEntry;

      ringEntry_t();

      void removeRef();
      void dontUseRefs();

      bool isAlone() const;
    };

    template <class entry_t>
    class ring_t {
    public:
      bool useRefs;
      ringEntry_t *head;

      ring_t();

      void dontUseRefs();
      void clear();

      void addRef(entry_t *entry);
      void removeRef(entry_t *entry);

      bool needsFree() const;
    };

    template <class entry_t>
    class multiRing_t {
    public:
      typedef ring_t<entry_t> entryRing_t;
      typedef std::map<entry_t*, entryRing_t> entryRingMap_t;

      bool useRefs;
      entryRingMap_t rings;

      multiRing_t();

      void dontUseRefs();
      void clear();

      void addNewRef(entry_t *entry);
      void removeRef(entry_t *entry);

      bool needsFree() const;
    };
  }
}

#include "gc.tpp"

#endif
