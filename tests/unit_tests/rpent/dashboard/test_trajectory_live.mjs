// Copyright 2026 The RPent Authors. Licensed under the Apache License, Version 2.0.
// Run with: node --test tests/unit_tests/rpent/dashboard/test_trajectory_live.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const script = readFileSync(new URL("../../../../rpent/dashboard/static/trajectory_live.js", import.meta.url), "utf8");
const flush = () => new Promise(resolve => setImmediate(resolve));

function harness() {
  const elements = new Map();
  const requests = [];
  const viewers = [];
  let source;
  const element = () => ({ textContent: "", children: [], append(child) { this.children.push(child); }, replaceChildren() { this.children = []; } });
  class Viewer {
    constructor(root, adapter) { this.adapter = adapter; this.updates = []; viewers.push(this); }
    update(data) { this.updates.push(data); }
  }
  vm.runInNewContext(script, {
    document: { documentElement: { lang: "en" }, getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); }, createElement: element },
    window: { RPentTrajectory: Viewer, addEventListener() {} },
    EventSource: class { constructor() { source = this; } close() {} },
    fetch(url, options) { return new Promise(resolve => { requests.push({ url, options, resolve }); }); },
    AbortController, DOMException, URLSearchParams,
    setInterval() { throw new Error("live trajectory must follow SSE, not polling"); },
  });
  return {
    requests, viewers,
    snapshot(runId, generation, seq) { source.onmessage({ data: JSON.stringify({ task_generation: generation, trace: { run_id: runId, last_seq: seq } }) }); },
    reconnect() { source.onopen(); },
    async respond(index, runId, generation, seq, extra = {}) {
      requests[index].resolve({ ok: true, json: async () => ({ task_generation: generation, summary: { run_id: runId, last_seq: seq }, ...extra }) });
      await flush();
    },
  };
}

test("coalesces SSE events and reconnects from the latest consumed sequence", async () => {
  const app = harness();
  app.snapshot("run-1", 1, 1);
  app.snapshot("run-1", 1, 2);
  assert.equal(app.requests.length, 1);
  assert.match(app.requests[0].url, /after_seq=0/);
  await app.respond(0, "run-1", 1, 1);
  assert.equal(app.requests.length, 2);
  assert.match(app.requests[1].url, /after_seq=1/);
  await app.respond(1, "run-1", 1, 2);
  app.reconnect();
  assert.equal(app.requests.length, 3);
  assert.match(app.requests[2].url, /after_seq=2/);
  await app.respond(2, "run-1", 1, 4);
  assert.deepEqual(app.viewers[0].updates.map(data => data.summary.last_seq), [1, 2, 4]);
});

test("a stale index cannot overwrite the view after a run or task switch", async () => {
  const app = harness();
  app.snapshot("old", 1, 1);
  app.snapshot("new", 2, 1);
  assert.equal(app.requests[0].options.signal.aborted, true);
  await app.respond(1, "new", 2, 1);
  await app.respond(0, "old", 1, 9);
  assert.equal(app.viewers[0].updates.length, 0);
  assert.equal(app.viewers[1].updates.length, 1);
  assert.equal(app.viewers[1].updates[0].summary.run_id, "new");
});

test("stale details are rejected and media URLs stay bound to their run", async () => {
  const app = harness();
  app.snapshot("old", 1, 1);
  await app.respond(0, "old", 1, 1);
  const old = app.viewers[0];
  const media = new URL(old.adapter.resolveMedia("trace/content/a b.png"), "http://localhost");
  assert.equal(media.searchParams.get("run_id"), "old");
  assert.equal(media.searchParams.get("generation"), "1");
  assert.equal(media.searchParams.get("ref"), "trace/content/a b.png");
  const detail = old.adapter.loadTurn("turn-1");
  const rejected = assert.rejects(detail, error => error.name === "AbortError");
  app.snapshot("new", 1, 1);
  assert.equal(old.adapter.resolveMedia("trace/content/a b.png"), null);
  await app.respond(1, "old", 1, 1, { run_id: "old", turn_id: "turn-1" });
  await rejected;
  await app.respond(2, "new", 1, 1);
  assert.equal(app.viewers[1].updates.length, 1);
});
