'use client';

import { CheckCircle2, Clock, XCircle, AlertCircle, ExternalLink, Box } from 'lucide-react';
import { clsx } from 'clsx';

const statusConfig: any = {
    PENDING: { color: 'text-yellow-400', icon: Clock, label: 'Pending Worker' },
    RUNNING: { color: 'text-blue-400', icon: AlertCircle, label: 'In Progress' },
    COMPLETED: { color: 'text-accent', icon: CheckCircle2, label: 'Ready' },
    FAILED: { color: 'text-red-400', icon: XCircle, label: 'Failed' },
};

export function JobCard({ job }: { job: any }) {
    const config = statusConfig[job.status] || statusConfig.PENDING;
    const Icon = config.icon;

    return (
        <div className="glass p-6 rounded-2xl flex items-center justify-between border-l-4" style={{ borderColor: `var(--${job.status === 'COMPLETED' ? 'accent' : job.status === 'FAILED' ? 'red-400' : 'primary'})` }}>
            <div className="flex items-center gap-4">
                <div className={clsx("p-3 rounded-xl bg-white/5", config.color)}>
                    <Icon size={24} />
                </div>
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className={clsx("text-xs font-bold px-2 py-0.5 rounded-full bg-white/5 uppercase tracking-wider", config.color)}>
                            {config.label}
                        </span>
                        <span className="text-xs text-gray-500">#{job.id.slice(0, 8)}</span>
                    </div>
                    <p className="text-sm font-medium text-gray-300">{job.message || 'Processing scan...'}</p>
                </div>
            </div>

            <div className="flex gap-3">
                {job.result_url && (
                    <a
                        href={job.result_url}
                        target="_blank"
                        className="glass p-2 rounded-lg hover:bg-white/10 transition-colors"
                        title="Download 3D Result"
                    >
                        <Box size={20} className="text-accent" />
                    </a>
                )}
                <a
                    href={job.video_url}
                    target="_blank"
                    className="glass p-2 rounded-lg hover:bg-white/10 transition-colors"
                    title="View Source Video"
                >
                    <ExternalLink size={20} className="text-gray-400" />
                </a>
            </div>
        </div>
    );
}
