import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

function detectPythonCommand() {
  // Check virtual environment in root or backend
  const venvCandidates = [
    path.join(rootDir, '.venv', 'Scripts', 'python.exe'),
    path.join(rootDir, '.venv', 'bin', 'python'),
    path.join(rootDir, 'backend', '.venv', 'Scripts', 'python.exe'),
    path.join(rootDir, 'backend', '.venv', 'bin', 'python'),
  ];

  for (const venvPath of venvCandidates) {
    if (existsSync(venvPath)) {
      return { cmd: venvPath, args: [] };
    }
  }

  // System Python candidates: prefer the Python environment that has fastapi & shap installed
  const systemCandidates = process.platform === 'win32'
    ? ['python', 'py -3.11', 'py', 'python3']
    : ['python3', 'python', 'py'];

  for (const candidate of systemCandidates) {
    try {
      const parts = candidate.split(' ');
      const res = spawnSync(parts[0], [...parts.slice(1), '-c', 'import fastapi, shap'], { stdio: 'ignore', shell: true });
      if (res.status === 0) {
        return { cmd: candidate, args: [] };
      }
    } catch (_) {
      // Continue to next candidate
    }
  }

  for (const candidate of systemCandidates) {
    try {
      const parts = candidate.split(' ');
      const res = spawnSync(parts[0], [...parts.slice(1), '--version'], { stdio: 'ignore', shell: true });
      if (res.status === 0) {
        return { cmd: candidate, args: [] };
      }
    } catch (_) {
      // Continue to next candidate
    }
  }

  // Default fallback
  return { cmd: 'python', args: [] };
}

const { cmd } = detectPythonCommand();
const uvicornArgs = ['-m', 'uvicorn', 'backend.main:app', '--reload', '--host', '127.0.0.1', '--port', '8000'];

console.log(`[AquaGuard Backend] Starting FastAPI using '${cmd}'...`);

const child = spawn(cmd, uvicornArgs, {
  cwd: rootDir,
  stdio: 'inherit',
  shell: true,
});

child.on('error', (err) => {
  console.error('[AquaGuard Backend] Failed to launch server process:', err.message);
  process.exit(1);
});

child.on('exit', (code, signal) => {
  if (signal) {
    process.exit(0);
  }
  process.exit(code ?? 0);
});

for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(sig, () => {
    if (!child.killed) {
      try {
        child.kill(sig);
      } catch (_) {}
    }
  });
}
