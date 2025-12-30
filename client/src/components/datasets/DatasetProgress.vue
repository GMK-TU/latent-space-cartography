<template>
  <div class="ds-progress" v-if="show">
    <div class="ds-progress__row">
      <div class="ds-progress__bar">
        <div class="ds-progress__fill" :style="{ width: clamped + '%' }"></div>
      </div>
      <div class="ds-progress__pct">{{ clamped }}%</div>
    </div>
    <div class="ds-progress__msg" v-if="message">{{ message }}</div>
  </div>
</template>

<script>
export default {
  name: "DatasetProgress",
  props: {
    progress: { type: Number, default: 0 },
    message: { type: String, default: "" },
    show: { type: Boolean, default: true },
  },
  computed: {
    clamped() {
      const p = Number(this.progress || 0);
      return Math.max(0, Math.min(100, Math.round(p)));
    },
  },
};
</script>

<style scoped>
.ds-progress__row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ds-progress__bar {
  flex: 1;
  height: 8px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(0,0,0,0.1);
}
.ds-progress__fill {
  height: 100%;
  background: rgba(0,0,0,0.35);
}
.ds-progress__pct {
  min-width: 44px;
  text-align: right;
  font-size: 12px;
  opacity: 0.8;
}
.ds-progress__msg {
  margin-top: 4px;
  font-size: 12px;
  opacity: 0.85;
}
</style>
