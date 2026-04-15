'use client';

import { useState } from 'react';
import { 
    CheckCircle2, 
    Clock, 
    XCircle, 
    ExternalLink, 
    Activity, 
    Sparkles, 
    Download, 
    Play,
    Loader2
} from 'lucide-react';
import { clsx } from 'clsx';

const statusConfig: any = {
    PENDING: { color: 'text-yellow-400', icon: Clock, label: 'Pending Worker' },
    SFM_RUNNING: { color: 'text-blue-400', icon: Activity, label: 'SfM: Extracting Frames' },
    SFM_COMPLETED: { color: 'text-green-400', icon: CheckCircle2, label: 'SfM: Ready to Train' },
    ready_to_train: { color: 'text-green-400', icon: CheckCircle2, label: 'SfM: Ready to Train' }, // Backwards compatibility
    SFM_FAILED: { color: 'text-red-400', icon: XCircle, label: 'SfM: Failed' },
    TRAINING_RUNNING: { color: 'text-purple-400', icon: Sparkles, label: 'Training: Generating 3DGS' },
    TRAINING_FAILED: { color: 'text-red-400', icon: XCircle, label: 'Training: Failed' },
    COMPLETED: { color: 'text-accent', icon: CheckCircle2, label: 'Result Ready' },
    FAILED: { color: 'text-red-400', icon: XCircle, label: 'Job Failed' },
};

export function JobCard({ job }: { job: any }) {
    const [isTriggering, setIsTriggering] = useState(false);
    const config = statusConfig[job.status] || statusConfig.PENDING;
    const Icon = config.icon;

    const handleStartTraining = async () => {
        setIsTriggering(true);
        try {
            const res = await fetch('/api/run-worker', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    jobId: job.id,
                    videoUrl: job.video_url,
                    mode: 'TRAIN'
                })
            });
            if (!res.ok) throw new Error('Failed to trigger training');
            console.log('✅ Training triggered');
        } catch (err: any) {
            alert(err.message);
        } finally {
            setIsTriggering(false);
        }
    };

    const isCompleted = job.status === 'COMPLETED';
    const isSfmDone = job.status === 'SFM_COMPLETED' || job.status === 'ready_to_train';

    return (
        <div className="glass p-6 rounded-2xl flex items-center justify-between border-l-4 transition-all" 
             style={{ borderColor: isCompleted ? 'var(--accent)' : 'rgba(255,255,255,0.1)' }}>
            <div className="flex items-center gap-4">
                <div className={clsx("p-3 rounded-xl bg-white/5", config.color)}>
                    {isTriggering ? <Loader2 className="animate-spin" size={24} /> : <Icon size={24} />}
                </div>
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className={clsx("text-[10px] font-bold px-2 py-0.5 rounded-full bg-white/5 uppercase tracking-wider", config.color)}>
                            {config.label}
                        </span>
                        <span className="text-xs text-gray-500">#{job.id.slice(0, 8)}</span>
                    </div>
                    <p className="text-sm font-medium text-gray-300 max-w-md truncate">
                        {job.message && job.message.startsWith('S3_PATH:') ? 'Data ready on S3' : (job.message || 'Processing...')}
                    </p>
                </div>
            </div>

            <div className="flex gap-3">
                {isSfmDone && (
                    <button
                        onClick={handleStartTraining}
                        disabled={isTriggering}
                        className="flex items-center gap-2 bg-primary/20 hover:bg-primary/30 text-primary px-4 py-2 rounded-xl text-sm font-bold transition-all border border-primary/20"
                    >
                        <Play size={16} fill="currentColor" />
                        Start Training
                    </button>
                )}
                
                {job.result_url && (
                    <a
                        href={job.result_url}
                        target="_blank"
                        className="flex items-center gap-2 bg-accent text-black px-4 py-2 rounded-xl text-sm font-bold hover:scale-105 transition-all"
                        title="Download 3D Result"
                    >
                        <Download size={18} />
                        Download PLY
                    </a>
                )}
                
                <a
                    href={job.video_url}
                    target="_blank"
                    className="glass p-2.5 rounded-xl hover:bg-white/10 transition-colors"
                    title="View Source Video"
                >
                    <ExternalLink size={20} className="text-gray-400" />
                </a>
            </div>
        </div>
    );
}
