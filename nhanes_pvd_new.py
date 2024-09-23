import pickle
import pandas as pd
import os
import pySuStaIn
import matplotlib.pyplot as plt
import numpy as np



# Load necessary variables saved from script 1
with open('sustain_output_full.pkl', 'rb') as f:
    output_folder, dataset_name, N_S_max, N_iterations_MCMC, zdata, sustain_input = pickle.load(f)


# Directory where the pickle files are stored
pickle_dir = os.path.join(output_folder, 'pickle_files')

Z_vals  = np.array([[ 1, 5,9 ,13],
                 [ 2, 5, 10, 13],
                 [ 2, 5, 9, 12],
                 [ 1, 4, 8, 11],
                 [ 1, 3,  7, 10]]) 


M = len(zdata) 

# get the sample sequences and f
pickle_filename_s = "nhanes_1000pt_100000MCMC_10st_Output_subtype3.pickle"
pickle_filepath_s = os.path.join(pickle_dir, pickle_filename_s)
pk = pd.read_pickle(pickle_filepath_s)  # Load the pickle file
print(pk)
samples_sequence = pk["samples_sequence"]
samples_f = pk["samples_f"]

def plot_positional_var(samples_sequence, samples_f, n_samples, Z_vals, biomarker_labels=None, ml_f_EM=None, cval=False, subtype_order=None, biomarker_order=None, title_font_size=12, stage_font_size=10, stage_label='SuStaIn Stage', stage_rot=0, stage_interval=1, label_font_size=10, label_rot=0, cmap="original", biomarker_colours=None, figsize=None, subtype_titles=None, separate_subtypes=False, save_path=None, save_kwargs={}):
    
    # Get the number of subtypes
    N_S = samples_sequence.shape[0]
    
    # Get the number of features/biomarkers
    N_bio = Z_vals.shape[0]
    
    # Check that the number of labels given matches
    if biomarker_labels is not None:
        assert len(biomarker_labels) == N_bio
    
    # Set subtype order if not given
    if subtype_order is None:
        # Determine order if info is given
        if ml_f_EM is not None:
            subtype_order = np.argsort(ml_f_EM)[::-1]
        else:
            subtype_order = np.argsort(np.mean(samples_f, 1))[::-1]
    elif isinstance(subtype_order, tuple):
        subtype_order = list(subtype_order)

    # Create a color matrix with specific, consistent colors for each z-score (stage)
    colour_mat = np.array([[1, 0, 0],  # Red for stage 1
                           [0, 1, 0],  # Green for stage 2
                           [0, 0, 1],  # Blue for stage 3
                           [1, 1, 0]]) # Yellow for stage 4
    
    N_z = Z_vals.shape[1]  # Number of z-scores (stages)
    
    # Flatten the Z_vals matrix to get each biomarker's stage progression
    stage_zscore = np.repeat(np.arange(N_z), N_bio)
    
    # Extract which biomarkers have which z-scores/stages (we'll map by index position now)
    stage_biomarker_index = np.tile(np.arange(N_bio), (N_z,))

    # Warn user of reordering if labels and order given
    if biomarker_labels is not None and biomarker_order is not None:
        warnings.warn("Both labels and an order have been given. The labels will be reordered according to the given order!")
    
    if biomarker_order is not None:
        biomarker_order = np.arange(N_bio) if len(biomarker_order) > N_bio else biomarker_order
    else:
        biomarker_order = np.arange(N_bio)
    
    # If no labels given, set dummy defaults
    if biomarker_labels is None:
        biomarker_labels = [f"Biomarker {i}" for i in range(N_bio)]
    else:
        biomarker_labels = [biomarker_labels[i] for i in biomarker_order]

    # Check number of subtype titles is correct if given
    if subtype_titles is not None:
        assert len(subtype_titles) == N_S

    # Check biomarker label colours
    if biomarker_colours is not None:
        biomarker_colours = AbstractSustain.check_biomarker_colours(biomarker_colours, biomarker_labels)
    else:
        biomarker_colours = {i: "black" for i in biomarker_labels}

    # Flag to plot subtypes separately
    if separate_subtypes:
        nrows, ncols = 1, 1
    else:
        if N_S == 1:
            nrows, ncols = 1, 1
        elif N_S < 3:
            nrows, ncols = 1, N_S
        elif N_S < 7:
            nrows, ncols = 2, int(np.ceil(N_S / 2))
        else:
            nrows, ncols = 3, int(np.ceil(N_S / 3))
    
    # Total axes used to loop over
    total_axes = nrows * ncols
    subtype_loops = N_S if separate_subtypes else 1
    figs = []
    
    # Loop over figures (only makes a diff if separate_subtypes=True)
    for i in range(subtype_loops):
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        figs.append(fig)

        for j in range(total_axes):
            if not separate_subtypes:
                i = j
            if isinstance(axs, np.ndarray):
                ax = axs.flat[i]
            else:
                ax = axs
            if i not in range(N_S):
                ax.set_axis_off()
                continue

            this_samples_sequence = samples_sequence[subtype_order[i], :, :].T
            N = this_samples_sequence.shape[1]

            # Construct confusion matrix (vectorized)
            confus_matrix = (this_samples_sequence == np.arange(N)[:, None, None]).sum(1) / this_samples_sequence.shape[0]

            confus_matrix_c = np.ones((N_bio, N, 3))

            # Assign colors based on the index of the z-score, ensuring consistent color order for each biomarker
            for j in range(N_z):
                alter_level = colour_mat[j] == 0
                confus_matrix_zscore = confus_matrix[stage_zscore == j]
                confus_matrix_c[
                    np.ix_(stage_biomarker_index[stage_zscore == j], range(N), alter_level)
                ] -= np.tile(
                    confus_matrix_zscore.reshape((stage_zscore == j).sum(), N, 1),
                    (1, 1, alter_level.sum())
                )

            if subtype_titles is not None:
                title_i = subtype_titles[i]
            else:
                temp_mean_f = np.mean(samples_f, 1)
                vals = temp_mean_f[subtype_order]
                if n_samples != np.inf:
                    title_i = f"Subtype {i+1} (f={vals[i]:.2f}, n={np.round(vals[i] * n_samples):n})"
                else:
                    title_i = f"Subtype {i+1} (f={vals[i]:.2f})"
            ax.imshow(confus_matrix_c[biomarker_order, :, :], interpolation='nearest')

            stage_ticks = np.arange(0, N, stage_interval)
            ax.set_xticks(stage_ticks)
            ax.set_xticklabels(stage_ticks + 1, fontsize=stage_font_size, rotation=stage_rot)

            ax.set_yticks(np.arange(N_bio))
            if (i % ncols) == 0:
                ax.set_yticklabels(biomarker_labels, ha="right", fontsize=label_font_size, rotation=label_rot)
                for tick_label in ax.get_yticklabels():
                    tick_label.set_color(biomarker_colours[tick_label.get_text()])
            else:
                ax.set_yticklabels([])

            ax.set_xlabel(stage_label, fontsize=stage_font_size + 2)
            ax.set_title(title_i, fontsize=title_font_size)

        fig.tight_layout()
        if save_path is not None:
            save_name = f"{save_path}_subtype{i}" if separate_subtypes else f"{save_path}_all-subtypes"
            file_format = save_kwargs.pop("format", "png")
            fig.savefig(f"{save_name}.{file_format}", **save_kwargs)
    
    return figs, axs

plot_positional_var(samples_sequence, samples_f, M, Z_vals)


# Access the current figure and modify it
fig = plt.gcf()  # Get the current figure
fig.set_size_inches(16, 12)  # Resize the figure

# Adjust axis ticks and labels for all subplots
plt.xticks(fontsize=8, rotation=45)  # Adjust x-tick labels
plt.yticks(fontsize=8)  # Adjust y-tick labels
plt.tight_layout()  # Adjust layout to prevent overlap

# Save the figure
fig.savefig('/gae/sustain/sustain_troubleshooting/nhanes_pilot/nhanes_1000pt_100000MCMC_10st_Output/figures/positional_variance_diagrams_fixed.png', dpi=300)

plt.close()