WITH hard_runs AS (
    SELECT *
    FROM public.hard_quiz_generation_runs
    WHERE difficulty = 'hard'
    -- Add a created_at range here to isolate a specific test batch.
),
summary AS (
    SELECT
        count(*) AS total_hard_quizzes_generated,
        avg(acceptance_rate) AS average_acceptance_rate,
        percentile_cont(0.5)
            WITHIN GROUP (ORDER BY acceptance_rate)
            AS median_acceptance_rate,
        min(acceptance_rate) AS min_acceptance_rate,
        max(acceptance_rate) AS max_acceptance_rate,
        avg(generated_questions) AS average_generated_questions,
        avg(accepted_questions) AS average_accepted_questions,
        avg(rejected_questions) AS average_rejected_questions
    FROM hard_runs
),
total_rejections AS (
    SELECT COALESCE(sum(rejected_questions), 0) AS count
    FROM hard_runs
),
reason_counts AS (
    SELECT
        reason.key AS rejection_reason,
        sum(reason.value::integer) AS rejection_count
    FROM hard_runs
    CROSS JOIN LATERAL jsonb_each_text(
        rejection_reasons_breakdown
    ) AS reason
    GROUP BY reason.key
)
SELECT row_to_json(summary) AS hard_generation_summary
FROM summary;

WITH hard_runs AS (
    SELECT *
    FROM public.hard_quiz_generation_runs
    WHERE difficulty = 'hard'
    -- Use the same created_at range as the summary query.
),
total_rejections AS (
    SELECT COALESCE(sum(rejected_questions), 0) AS count
    FROM hard_runs
),
reason_counts AS (
    SELECT
        reason.key AS rejection_reason,
        sum(reason.value::integer) AS rejection_count
    FROM hard_runs
    CROSS JOIN LATERAL jsonb_each_text(
        rejection_reasons_breakdown
    ) AS reason
    GROUP BY reason.key
)
SELECT
    rejection_reason,
    rejection_count,
    CASE
        WHEN total_rejections.count = 0 THEN 0
        ELSE round(
            100.0 * rejection_count / total_rejections.count,
            2
        )
    END AS percentage_of_total_rejections
FROM reason_counts
CROSS JOIN total_rejections
ORDER BY rejection_count DESC, rejection_reason;
