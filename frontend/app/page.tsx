'use client';

import { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabase';
import { UploadZone } from '@/components/UploadZone';
import { JobCard } from '@/components/JobCard';
import { Activity, Database, Sparkles } from 'lucide-react';

export default function Home() {
  const [jobs, setJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchJobs();

    // Realtime subscription
    const channel = supabase
      .channel('public:jobs')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'jobs' }, () => {
        fetchJobs();
      })
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  const fetchJobs = async () => {
    const { data, error } = await supabase
      .from('jobs')
      .select('*')
      .order('created_at', { ascending: false });

    if (error) console.error('Error fetching jobs:', error);
    else setJobs(data || []);
    setLoading(false);
  };

  return (
    <main className="min-h-screen p-8 max-w-6xl mx-auto">
      <header className="flex justify-between items-center mb-12">
        <div>
          <h1 className="text-4xl font-bold mb-2 gradient-text">3D Scan & Gaussian Splatting</h1>
          <p className="text-gray-400">Convert Videos to High-Precision 3D Models</p>
        </div>
        <div className="flex gap-4">
          <div className="glass px-4 py-2 rounded-xl flex items-center gap-2">
            <Database size={18} className="text-blue-400" />
            <span className="text-sm font-medium">Supabase Linked</span>
          </div>
          <div className="glass px-4 py-2 rounded-xl flex items-center gap-2">
            <Sparkles size={18} className="text-yellow-400" />
            <span className="text-sm font-medium">Cloud Run Ready</span>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1">
          <UploadZone onUploadSuccess={fetchJobs} />
        </div>

        <div className="lg:col-span-2">
          <div className="flex items-center gap-2 mb-6">
            <Activity className="text-accent" />
            <h2 className="text-2xl font-semibold">Active Pipeline Jobs</h2>
          </div>

          <div className="space-y-4">
            {loading ? (
              <div className="glass p-8 rounded-2xl text-center animate-pulse">Loading jobs...</div>
            ) : jobs.length === 0 ? (
              <div className="glass p-12 rounded-2xl text-center opacity-60">
                No jobs found. Upload a video to start the process.
              </div>
            ) : (
              jobs.map((job) => <JobCard key={job.id} job={job} />)
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
