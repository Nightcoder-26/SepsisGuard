/**
 * SepsisGuard v3.0 - Web Audio API Medical Telemetry Beeping Engine
 * Generates real-time synthetic medical telemetry warning & critical alarm sound patterns
 * without external audio file dependencies.
 */

class TelemetryAudioEngine {
    constructor() {
        this.ctx = null;
        this.enabled = true;
        this.initialized = false;
        
        // Auto-initialize audio context on first user interaction anywhere on page
        const unlock = () => {
            this.init();
            document.removeEventListener('click', unlock);
            document.removeEventListener('keydown', unlock);
            document.removeEventListener('touchstart', unlock);
        };
        document.addEventListener('click', unlock, { passive: true });
        document.addEventListener('keydown', unlock, { passive: true });
        document.addEventListener('touchstart', unlock, { passive: true });
    }

    init() {
        if (!this.ctx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                this.ctx = new AudioCtx();
                this.initialized = true;
            }
        }
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    toggleSound() {
        this.init();
        this.enabled = !this.enabled;
        this.updateButtons();
        if (this.enabled) {
            this.playBeep('WARNING');
        }
        return this.enabled;
    }

    updateButtons() {
        const btns = document.querySelectorAll('.audio-toggle-btn');
        btns.forEach(btn => {
            if (this.enabled) {
                btn.innerHTML = '<span>🔊</span><span>AUDIO: ON</span>';
                btn.classList.remove('muted');
            } else {
                btn.innerHTML = '<span>🔇</span><span>AUDIO: MUTED</span>';
                btn.classList.add('muted');
            }
        });
    }

    playBeep(level) {
        if (!this.enabled) return;
        this.init();
        if (!this.ctx) return;

        try {
            const now = this.ctx.currentTime;
            
            if (level === 'WARNING' || level === 'Medium') {
                // WARNING BEEP: Dual-tone medium medical pulse (660Hz -> 550Hz, 300ms duration)
                const osc = this.ctx.createOscillator();
                const gain = this.ctx.createGain();
                
                osc.type = 'sine';
                osc.frequency.setValueAtTime(660, now);
                osc.frequency.exponentialRampToValueAtTime(550, now + 0.15);

                gain.gain.setValueAtTime(0.2, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.28);

                osc.connect(gain);
                gain.connect(this.ctx.destination);

                osc.start(now);
                osc.stop(now + 0.3);
            } 
            else if (level === 'CRITICAL' || level === 'High') {
                // CRITICAL ALARM BEEP: Triple high-urgency medical pulse (880Hz / 1040Hz / 880Hz rapid alert)
                [0, 0.12, 0.24].forEach((delay, idx) => {
                    const osc = this.ctx.createOscillator();
                    const gain = this.ctx.createGain();
                    
                    osc.type = 'square';
                    const freq = idx === 1 ? 1040 : 880;
                    osc.frequency.setValueAtTime(freq, now + delay);

                    gain.gain.setValueAtTime(0.18, now + delay);
                    gain.gain.exponentialRampToValueAtTime(0.001, now + delay + 0.09);

                    osc.connect(gain);
                    gain.connect(this.ctx.destination);

                    osc.start(now + delay);
                    osc.stop(now + delay + 0.1);
                });
            }
        } catch (e) {
            console.warn('[Audio Engine] Playback failed:', e);
        }
    }
}

// Global Singleton Instance
window.telemetryAudio = new TelemetryAudioEngine();
