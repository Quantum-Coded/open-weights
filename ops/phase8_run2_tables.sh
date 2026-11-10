cd ~/wsp-r1 && source .venv/bin/activate
{
echo "## MATH-500/200 (vs base)"; python analyze.py --base results/base_math500_greedy.jsonl --arm results/edit_r2_math500_greedy.jsonl --arm results/rand_w_r2_math500_greedy.jsonl --arm results/rand_s_r2_math500_greedy.jsonl --arm results/edit_math500_greedy.jsonl --arm results/base_pp_math500_greedy.jsonl --arm results/edit_pp_r2_math500_greedy.jsonl --arm results/edit_rerun_r2_math500_greedy.jsonl 2>&1 | grep -v "^$"
echo "## GSM8K test400"; python analyze.py --base results/base_gsm8k_test400_greedy.jsonl --arm results/edit_r2_gsm8k_test400_greedy.jsonl --arm results/edit_gsm8k_test400_greedy.jsonl 2>&1 | grep "|"
echo "## MMLU-Pro300"; python analyze.py --base results/base_mmlupro300_greedy.jsonl --arm results/edit_r2_mmlupro300_greedy.jsonl --arm results/edit_mmlupro300_greedy.jsonl 2>&1 | grep "|"
echo "## gsm8k_fresh400"; python analyze.py --base results/base_gsm8k_fresh400_greedy.jsonl --arm results/e2_gsm8k_fresh400_greedy.jsonl --arm results/e1_gsm8k_fresh400_greedy.jsonl --arm results/rand_w_gsm8k_fresh400_greedy.jsonl 2>&1 | grep "|"
echo "## mmlupro_fresh300"; python analyze.py --base results/base_mmlupro_fresh300_greedy.jsonl --arm results/e2_mmlupro_fresh300_greedy.jsonl --arm results/e1_mmlupro_fresh300_greedy.jsonl --arm results/rand_w_mmlupro_fresh300_greedy.jsonl 2>&1 | grep "|"
echo "## full MATH-500"; python analyze.py --base results/base_full_math500_greedy.jsonl --arm results/edit_full_r2_math500_greedy.jsonl --arm results/edit_full_math500_greedy.jsonl 2>&1 | grep "|"
echo "## sampled"; python analyze.py --base results/base_math500_sampled.jsonl --arm results/edit_r2_math500_sampled.jsonl --arm results/edit_math500_sampled.jsonl 2>&1 | grep "|"
} > logs_phase8_run2_tables.txt 2>&1
ls results | grep -E "r2|full" ; cat logs_phase8_run2_tables.txt
