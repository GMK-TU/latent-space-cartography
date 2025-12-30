<template>
  <div class="ds-preview">
    <div v-if="images && images.length">
      <div class="ds-preview__title">Image preview</div>
      <div class="ds-preview__grid">
        <img v-for="(src, i) in images" :key="i" :src="src" class="ds-preview__img" />
      </div>
    </div>

    <div v-if="meta && meta.length" class="ds-preview__meta">
      <div class="ds-preview__title">Metadata preview</div>
      <pre class="ds-preview__pre">{{ prettyMeta }}</pre>
    </div>

    <div v-if="matched && matched.length" class="ds-preview__matched">
      <div class="ds-preview__title">Matched preview</div>
      <div class="ds-preview__matchedRow" v-for="(row, i) in matched" :key="i">
        <img v-if="row.imageUrl" :src="row.imageUrl" class="ds-preview__imgSmall" />
        <pre class="ds-preview__preSmall">{{ pretty(row.metaRow || row.meta || row) }}</pre>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "DatasetPreview",
  props: {
    images: { type: Array, default: () => [] },
    meta: { type: Array, default: () => [] },
    matched: { type: Array, default: () => [] },
  },
  computed: {
    prettyMeta() {
      return this.pretty(this.meta);
    },
  },
  methods: {
    pretty(x) {
      try { return JSON.stringify(x, null, 2); } catch (_) { return String(x); }
    },
  },
};
</script>

<style scoped>
.ds-preview { display: grid; gap: 14px; }
.ds-preview__title { font-weight: 600; font-size: 13px; margin-bottom: 6px; }
.ds-preview__grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 8px; }
.ds-preview__img { width: 100%; height: 90px; object-fit: cover; border-radius: 8px; border: 1px solid rgba(0,0,0,0.12); }
.ds-preview__meta, .ds-preview__matched { }
.ds-preview__pre { background: rgba(0,0,0,0.04); padding: 10px; border-radius: 8px; overflow: auto; max-height: 220px; }
.ds-preview__matchedRow { display: flex; gap: 10px; align-items: flex-start; }
.ds-preview__imgSmall { width: 64px; height: 64px; object-fit: cover; border-radius: 8px; border: 1px solid rgba(0,0,0,0.12); }
.ds-preview__preSmall { flex: 1; background: rgba(0,0,0,0.04); padding: 8px; border-radius: 8px; overflow: auto; max-height: 140px; margin: 0; }
</style>
