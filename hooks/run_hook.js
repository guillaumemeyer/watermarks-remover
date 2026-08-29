#!/usr/bin/env node
/**
 * Cross-platform launcher for the watermarks-remover Claude Code plugin hook.
 *
 * Claude Code runs on Node.js across Windows, macOS, and Linux, so `node` is
 * always available. Python, however, may be `py -3`, `python3`, or `python`
 * depending on platform and installation. This runner probes each candidate to
 * confirm it exists and is a Python 3.10+ interpreter (the repository's floor)
 * before committing to it, then proxies stdin/stdout/stderr and the exit code
 * straight through to hook_written_file.py.
 */

const { spawn, execFile } = require('child_process');
const path = require('path');

const script = path.resolve(__dirname, '..', 'service', 'scripts', 'hook_written_file.py');

const isWin = process.platform === 'win32';
const candidates = isWin
  ? [
      ['py', ['-3']],
      ['python', []],
      ['python3', []],
    ]
  : [
      ['python3', []],
      ['python', []],
    ];

// Repo requirement: Python 3.10+ stdlib only. `python` may be Python 2 or an
// older Python 3, so the probe must reject anything below this floor.
const VERSION_CHECK = 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)';

// The interpreter we committed to running. Guarding on this means a candidate
// that failed to spawn (ENOENT) cannot let its later `close` event terminate
// the process out from under a fallback candidate already running.
let active = null;

function run(cmd, extraArgs, index) {
  const args = [...extraArgs, script, ...process.argv.slice(2)];
  const child = spawn(cmd, args, { stdio: 'inherit', windowsHide: true });
  active = child;

  child.on('error', (err) => {
    if (err.code === 'ENOENT') {
      // The interpreter vanished between probe and launch; fall back rather
      // than die. Clearing active makes this child's later close a no-op.
      active = null;
      launch(index + 1);
    } else {
      console.error(`watermarks-remover: error running ${cmd}: ${err.message}`);
      process.exit(1);
    }
  });

  child.on('close', (code) => {
    // Only the selected child may exit the process; a failed candidate that
    // still fires close after an ENOENT error must not kill a fallback.
    if (active === child) process.exit(code ?? 0);
  });
}

function launch(index) {
  if (index >= candidates.length) {
    const tried = candidates
      .map(([cmd, extra]) => (extra.length ? `${cmd} ${extra.join(' ')}` : cmd))
      .join(', ');
    console.error(
      `watermarks-remover: could not find a working Python 3.10+ interpreter (tried ${tried})`
    );
    process.exit(1);
  }

  const [cmd, extraArgs] = candidates[index];

  // Probe the candidate: confirm it exists and is at least Python 3.10. Any
  // failure -- a missing binary or an older Python -- falls through to the next
  // candidate.
  execFile(cmd, [...extraArgs, '-c', VERSION_CHECK], (err) => {
    if (err) {
      launch(index + 1);
    } else {
      run(cmd, extraArgs, index);
    }
  });
}

launch(0);
