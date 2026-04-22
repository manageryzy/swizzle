import { render } from 'preact';
import { effect } from '@preact/signals';
import { App } from './App';
import { hydrateFromUrl, installUrlSync } from './url_state';
import { inst, resetTickIfInstChanged } from './state';

hydrateFromUrl();
installUrlSync();

// Reset tick on instruction switch so we don't scrub into a dead phase.
effect(() => {
  resetTickIfInstChanged(inst.value.id);
});

const root = document.getElementById('app');
if (root) render(<App />, root);
