





class TBLogger():
    def __init__(self, name, writer):
        self.name = name;
        self.writer = writer;
        self.training_prefix = 'train';
        self.testing_prefix = 'test';
        self.step = 0;


    def write_log_probs(self, name, log_probs):
        self.writer.add_scalars('%s/LogProb' % name,
                                log_probs,
                                self.step)


    def write_klds(self, name, klds):
        self.writer.add_scalars('%s/KLD' % name,
                                klds,
                                self.step)


    def write_group_div(self, name, group_div):
        self.writer.add_scalars('%s/group_divergence' % name,
                                {'joint_div':group_div},
                                self.step)

    def write_latent_distr(self, name, latents, step=None):
        write_step = self.get_step(step)
        l_mods = latents['modalities'];
        for k, key in enumerate(l_mods.keys()):
            if not l_mods[key][0] is None:
                self.writer.add_scalars('%s/mu' % name,
                                        {key: l_mods[key][0].mean().item()},
                                        write_step)
            if not l_mods[key][1] is None:
                self.writer.add_scalars('%s/logvar' % name,
                                        {key: l_mods[key][1].mean().item()},
                                        write_step)


    def write_lr_eval(self, lr_eval, step=None):
        write_step = self.get_step(step)
        self.writer.add_scalars('Latent_classification',
                                    lr_eval,
                                    write_step)


    def write_coherence_logs_subsets(self, gen_eval):
        for j, l_key in enumerate(sorted(gen_eval['cond'].keys())):
            for k, s_key in enumerate(gen_eval['cond'][l_key].keys()):
                self.writer.add_scalars('Generation/%s/%s' %
                                        (l_key, s_key),
                                        gen_eval['cond'][l_key][s_key],
                                        self.step)
        self.writer.add_scalars('Generation/Random',
                                gen_eval['random'],
                                self.step)

    def write_coherence_logs_all(self, gen_eval, step=None):
        write_step = self.get_step(step)
        for j, l_key in enumerate(sorted(gen_eval['cond'].keys())):
            self.writer.add_scalars('Coherence/generation/%s' %
                                    (l_key),
                                    gen_eval['cond'][l_key],
                                    write_step)
        self.writer.add_scalars('Coherence/Random',
                                gen_eval['random'],
                                write_step)
    
    def write_mse_all(self, gen_eval, step=None):
        write_step = self.get_step(step)
        for j, l_key in enumerate(sorted(gen_eval.keys())):
            self.writer.add_scalars('MSE/%s' %(l_key),
                                    gen_eval[l_key],
                                    write_step)

    def write_lhood_logs(self, lhoods):
        for k, key in enumerate(sorted(lhoods.keys())):
            self.writer.add_scalars('Likelihoods/%s'%
                                    (key),
                                    lhoods[key],
                                    self.step)
    
    def write_prd_scores(self, prd_scores):
        self.writer.add_scalars('PRD',
                                prd_scores,
                                self.step)


    def write_plots(self, plots, epoch):
        for k, p_key in enumerate(plots.keys()):
            ps = plots[p_key];
            for l, name in enumerate(ps.keys()):
                fig = ps[name];
                self.writer.add_image(p_key + '_' + name,
                                      fig,
                                      epoch,
                                      dataformats="HWC");



    def add_basic_logs(self, name, results, loss, log_probs, klds):
        self.writer.add_scalars('%s/Loss' % name,
                                loss,
                                self.step)
        self.write_log_probs(name, log_probs);
        self.write_klds(name, klds);
        self.write_group_div(name, results['joint_divergence'].item());
        self.write_latent_distr(name, results['latents']);


    def write_training_logs(self, results, loss, log_probs, klds):
        self.add_basic_logs(self.training_prefix, results, loss, log_probs, klds);
        self.step += 1;


    def write_testing_logs(self, results, loss, log_probs, klds):
        self.add_basic_logs(self.testing_prefix, results, loss, log_probs, klds);
        self.step += 1;

    def get_step(self, step):
        if type(step)!=type(None):
            write_step = step
        else:
            write_step = self.step
            self.step += 1
        return write_step

    def write_dict(self, scalars_dict, step=None):
        write_step = self.get_step(step)
            
        for k in scalars_dict.keys():
            self.writer.add_scalar(k, scalars_dict[k], write_step)

    def write_scalars(self, name, scalars_dict, step=None):
        write_step = self.get_step(step)
        self.writer.add_scalars(name, scalars_dict, write_step)
