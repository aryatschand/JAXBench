def process_seq(seq_idx):
      q_start = cu_q_lens[seq_idx]
      q_end = cu_q_lens[seq_idx + 1]
      q_len = q_end - q_start
      kv_len = kv_lens[seq_idx]
      indices = page_indices[seq_idx]
      
      # Gather KV for this sequence
      kv = kv_pages[indices] # (pages_per_seq, page_size, num_combined_kv_heads, D)
      
      # We need queries for this sequence.
      # Since q_len is dynamic, we can't easily pass a dynamic slice to Pallas.
      # BUT we can pass the WHOLE queries array to Pallas, and let Pallas slice it!
      # Wait, if we vmap over max_seqs, we can't pass the whole queries array to each vmap iteration without broadcasting it.
      # Broadcasting queries to (max_seqs, max_tokens, ...) is exactly what we want to avoid!
