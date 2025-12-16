"""
Agentic task execution module.
Implements intelligent task planning, execution, and coordination using retrieved knowledge.
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of agent tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskResult:
    """Result of a completed task."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentTask:
    """Represents an agent task."""
    task_id: str
    name: str
    description: str
    task_type: str  # 'analysis', 'generation', 'retrieval', 'synthesis'
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = None  # Task IDs this task depends on
    input_data: Dict[str, Any] = None
    context: List[Dict[str, Any]] = None  # Retrieved documents
    result: Optional[TaskResult] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.input_data is None:
            self.input_data = {}
        if self.context is None:
            self.context = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at


class TaskExecutor:
    """Executes individual agent tasks."""
    
    def __init__(self, gemini_client, embedding_manager, retriever):
        """
        Initialize task executor.
        
        Args:
            gemini_client: GeminiClient instance
            embedding_manager: EmbeddingManager instance
            retriever: AStarRetriever instance
        """
        self.gemini_client = gemini_client
        self.embedding_manager = embedding_manager
        self.retriever = retriever
        
        # Task execution functions
        self.task_functions = {
            'analysis': self._execute_analysis_task,
            'generation': self._execute_generation_task,
            'retrieval': self._execute_retrieval_task,
            'synthesis': self._execute_synthesis_task,
            'planning': self._execute_planning_task,
            'evaluation': self._execute_evaluation_task
        }
        
        # Task templates for common scenarios
        self.task_templates = self._initialize_task_templates()
        
        logger.info("Initialized TaskExecutor")
    
    def execute_task(self, task: AgentTask) -> TaskResult:
        """
        Execute a single agent task.
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult object
        """
        logger.info(f"Executing task: {task.name}")
        
        if task.task_type not in self.task_functions:
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"Unknown task type: {task.task_type}"
            )
        
        start_time = time.time()
        
        try:
            # Execute the task function
            result = self.task_functions[task.task_type](task)
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                metadata={
                    'task_type': task.task_type,
                    'priority': task.priority.value,
                    'dependencies': len(task.dependencies)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task execution failed: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={'task_type': task.task_type}
            )
    
    def _execute_analysis_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute analysis task."""
        prompt = task.input_data.get('prompt', task.description)
        context_texts = [ctx.get('text', '') for ctx in task.context]
        
        analysis_prompt = f"""
        Analyze the following information and provide detailed insights:
        
        Query: {prompt}
        
        Context Documents:
        {' '.join(context_texts)}
        
        Please provide:
        1. Key findings and insights
        2. Data patterns and trends
        3. Important relationships
        4. Potential implications
        5. Confidence assessment
        
        Structure your response as JSON with keys: findings, patterns, relationships, implications, confidence
        """
        
        response = self.gemini_client.generate(analysis_prompt, context_texts)
        
        try:
            # Try to parse as structured JSON
            analysis = json.loads(response.text)
            return {
                'type': 'analysis',
                'query': prompt,
                'context_count': len(context_texts),
                'analysis': analysis
            }
        except json.JSONDecodeError:
            return {
                'type': 'analysis',
                'query': prompt,
                'context_count': len(context_texts),
                'analysis': {
                    'raw_response': response.text,
                    'summary': 'Analysis completed but structured parsing failed'
                }
            }
    
    def _execute_generation_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute content generation task."""
        prompt = task.input_data.get('prompt', task.description)
        generation_type = task.input_data.get('generation_type', 'summary')
        context_texts = [ctx.get('text', '') for ctx in task.context]
        
        generation_prompt = f"""
        Generate {generation_type} based on the following information:
        
        Query: {prompt}
        
        Source Materials:
        {' '.join(context_texts)}
        
        Please create a comprehensive {generation_type} that is well-structured, informative, and directly answers the query.
        """
        
        response = self.gemini_client.generate(generation_prompt, context_texts)
        
        return {
            'type': 'generation',
            'generation_type': generation_type,
            'query': prompt,
            'content': response.text,
            'context_count': len(context_texts),
            'usage_metadata': response.usage
        }
    
    def _execute_retrieval_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute information retrieval task."""
        query = task.input_data.get('query', task.description)
        retrieval_type = task.input_data.get('retrieval_type', 'relevant_info')
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generator.generate_embedding(query)
        
        # Perform A* search
        search_results = self.retriever.search(query, query_embedding)
        
        return {
            'type': 'retrieval',
            'retrieval_type': retrieval_type,
            'query': query,
            'results_count': len(search_results),
            'results': search_results,
            'search_analytics': self.retriever.get_search_analytics()
        }
    
    def _execute_synthesis_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute synthesis task."""
        synthesis_prompt = task.input_data.get('prompt', task.description)
        synthesis_type = task.input_data.get('synthesis_type', 'comprehensive_review')
        
        # Collect all context from subtasks
        all_context = []
        for ctx_item in task.context:
            if isinstance(ctx_item, dict):
                all_context.append(str(ctx_item.get('text', '')))
            else:
                all_context.append(str(ctx_item))
        
        synthesis_query = f"""
        Synthesize information from multiple sources into a {synthesis_type}:
        
        Synthesis Request: {synthesis_prompt}
        
        Source Information:
        {' | '.join(all_context)}
        
        Provide a comprehensive synthesis that:
        1. Integrates information from all sources
        2. Identifies common themes and differences
        3. Provides balanced perspective
        4. Draws meaningful conclusions
        5. Suggests next steps or actions
        """
        
        response = self.gemini_client.generate(synthesis_query, all_context)
        
        return {
            'type': 'synthesis',
            'synthesis_type': synthesis_type,
            'prompt': synthesis_prompt,
            'sources_count': len(all_context),
            'synthesis': response.text,
            'source_summary': self._summarize_sources(all_context)
        }
    
    def _execute_planning_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute planning task."""
        planning_query = task.input_data.get('prompt', task.description)
        planning_type = task.input_data.get('planning_type', 'action_plan')
        
        planning_prompt = f"""
        Create a detailed {planning_type} based on the following requirements:
        
        Requirements: {planning_query}
        
        Context from documents:
        {' '.join([ctx.get('text', '') for ctx in task.context])}
        
        Your plan should include:
        1. Clear objectives and goals
        2. Step-by-step action items
        3. Timeline and milestones
        4. Resource requirements
        5. Risk assessment and mitigation
        6. Success metrics
        
        Format as structured JSON with clear sections.
        """
        
        response = self.gemini_client.generate(planning_prompt)
        
        try:
            plan = json.loads(response.text)
            return {
                'type': 'planning',
                'planning_type': planning_type,
                'query': planning_query,
                'plan': plan
            }
        except json.JSONDecodeError:
            return {
                'type': 'planning',
                'planning_type': planning_type,
                'query': planning_query,
                'plan': {
                    'raw_response': response.text,
                    'structured': False
                }
            }
    
    def _execute_evaluation_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute evaluation task."""
        evaluation_query = task.input_data.get('prompt', task.description)
        evaluation_criteria = task.input_data.get('criteria', ['relevance', 'completeness', 'accuracy'])
        
        # Get context to evaluate
        context_texts = [ctx.get('text', '') for ctx in task.context]
        
        evaluation_prompt = f"""
        Evaluate the following information based on these criteria: {', '.join(evaluation_criteria)}
        
        Query: {evaluation_query}
        
        Information to evaluate:
        {' '.join(context_texts)}
        
        Provide detailed evaluation with scores (1-10) for each criterion and overall assessment.
        Format as JSON with evaluation results and recommendations.
        """
        
        response = self.gemini_client.generate(evaluation_prompt, context_texts)
        
        try:
            evaluation = json.loads(response.text)
            return {
                'type': 'evaluation',
                'query': evaluation_query,
                'criteria': evaluation_criteria,
                'evaluation': evaluation,
                'overall_score': evaluation.get('overall_score', 'N/A')
            }
        except json.JSONDecodeError:
            return {
                'type': 'evaluation',
                'query': evaluation_query,
                'criteria': evaluation_criteria,
                'evaluation': {
                    'raw_response': response.text,
                    'evaluation_completed': True
                }
            }
    
    def _summarize_sources(self, sources: List[str]) -> Dict[str, Any]:
        """Summarize source materials."""
        if not sources:
            return {'count': 0, 'total_length': 0, 'types': []}
        
        return {
            'count': len(sources),
            'total_length': sum(len(s) for s in sources),
            'average_length': sum(len(s) for s in sources) / len(sources),
            'source_types': ['document'] * len(sources)  # Placeholder for source type detection
        }
    
    def _initialize_task_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize task templates for common scenarios."""
        return {
            'document_analysis': {
                'name': 'Document Analysis',
                'description': 'Analyze documents for key insights and themes',
                'task_type': 'analysis',
                'dependencies': [],
                'input_schema': {
                    'prompt': 'str',
                    'analysis_type': 'str'
                }
            },
            'research_synthesis': {
                'name': 'Research Synthesis',
                'description': 'Synthesize research findings from multiple sources',
                'task_type': 'synthesis',
                'dependencies': ['retrieval_task'],
                'input_schema': {
                    'prompt': 'str',
                    'synthesis_type': 'str'
                }
            },
            'action_planning': {
                'name': 'Action Planning',
                'description': 'Create detailed action plans based on research',
                'task_type': 'planning',
                'dependencies': ['analysis_task', 'synthesis_task'],
                'input_schema': {
                    'prompt': 'str',
                    'planning_type': 'str'
                }
            },
            'quality_evaluation': {
                'name': 'Quality Evaluation',
                'description': 'Evaluate the quality and completeness of information',
                'task_type': 'evaluation',
                'dependencies': ['synthesis_task'],
                'input_schema': {
                    'prompt': 'str',
                    'criteria': 'list'
                }
            },
            'content_generation': {
                'name': 'Content Generation',
                'description': 'Generate comprehensive content based on research',
                'task_type': 'generation',
                'dependencies': ['analysis_task', 'synthesis_task'],
                'input_schema': {
                    'prompt': 'str',
                    'generation_type': 'str'
                }
            }
        }


class AgentOrchestrator:
    """Orchestrates multiple agent tasks with dependency management."""
    
    def __init__(self, task_executor: TaskExecutor, max_concurrent_tasks: int = 4):
        """
        Initialize agent orchestrator.
        
        Args:
            task_executor: TaskExecutor instance
            max_concurrent_tasks: Maximum number of concurrent tasks
        """
        self.task_executor = task_executor
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Task registry
        self.tasks = {}
        self.task_results = {}
        self.task_lock = threading.RLock()
        
        # Execution statistics
        self.execution_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0
        }
        
        logger.info("Initialized AgentOrchestrator")
    
    def create_task_plan(self, user_query: str, context: List[Dict[str, Any]] = None) -> List[AgentTask]:
        """
        Create a task plan for a user query using agentic reasoning.
        
        Args:
            user_query: User's query or request
            context: Optional initial context documents
            
        Returns:
            List of tasks to execute
        """
        logger.info(f"Creating task plan for query: {user_query[:50]}...")
        
        # Analyze query complexity to determine task structure
        plan_prompt = f"""
        Analyze this user query and create a comprehensive task execution plan:
        
        Query: {user_query}
        
        Context: {str(context[:3]) if context else 'No initial context provided'}
        
        Consider:
        1. What type of task is this? (research, analysis, planning, generation)
        2. What are the key subtasks needed?
        3. What are the dependencies between tasks?
        4. What information needs to be retrieved?
        5. What final output is expected?
        
        Create a detailed plan with:
        - Task names and descriptions
        - Task types (retrieval, analysis, synthesis, planning, generation, evaluation)
        - Dependencies between tasks
        - Input requirements for each task
        
        Format as JSON with array of tasks.
        """
        
        try:
            # Use Gemini to generate the task plan
            response = self.task_executor.gemini_client.generate(plan_prompt)
            
            # Try to parse as JSON
            try:
                plan_data = json.loads(response.text)
                tasks = self._parse_plan_data(plan_data, user_query, context)
            except json.JSONDecodeError:
                # Fallback to a default plan structure
                tasks = self._create_default_plan(user_query, context)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Task plan creation failed: {e}")
            return self._create_default_plan(user_query, context)
    
    def execute_task_plan(self, tasks: List[AgentTask], timeout: int = 300) -> Dict[str, Any]:
        """
        Execute a task plan with dependency management.
        
        Args:
            tasks: List of tasks to execute
            timeout: Overall timeout in seconds
            
        Returns:
            Execution results dictionary
        """
        logger.info(f"Executing task plan with {len(tasks)} tasks")
        
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        
        # Register tasks
        for task in tasks:
            self.tasks[task.task_id] = task
        
        # Track task completion
        completed_tasks = set()
        failed_tasks = set()
        
        # Execute tasks in dependency order
        with ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as executor:
            while len(completed_tasks) + len(failed_tasks) < len(tasks):
                # Find tasks ready for execution (dependencies satisfied)
                ready_tasks = []
                for task in tasks:
                    if (task.task_id not in completed_tasks and 
                        task.task_id not in failed_tasks and
                        all(dep in completed_tasks for dep in task.dependencies)):
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    # Check for timeout
                    if time.time() - start_time > timeout:
                        logger.warning("Task execution timeout")
                        break
                    
                    # Wait a bit and try again
                    time.sleep(1)
                    continue
                
                # Execute ready tasks
                future_to_task = {
                    executor.submit(self._execute_single_task, task): task 
                    for task in ready_tasks
                }
                
                # Wait for tasks to complete
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        self.task_results[task.task_id] = result
                        
                        if result.status == TaskStatus.COMPLETED:
                            completed_tasks.add(task.task_id)
                            logger.info(f"Task completed: {task.name}")
                        else:
                            failed_tasks.add(task.task_id)
                            logger.error(f"Task failed: {task.name} - {result.error}")
                    
                    except Exception as e:
                        failed_tasks.add(task.task_id)
                        logger.error(f"Task execution exception: {task.name} - {e}")
            
            # Handle timeout - mark remaining tasks as failed
            for task in tasks:
                if task.task_id not in completed_tasks and task.task_id not in failed_tasks:
                    self.task_results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error="Execution timeout"
                    )
                    failed_tasks.add(task.task_id)
        
        # Compile final results
        execution_time = time.time() - start_time
        
        results = {
            'execution_id': execution_id,
            'execution_time': execution_time,
            'total_tasks': len(tasks),
            'completed_tasks': len(completed_tasks),
            'failed_tasks': len(failed_tasks),
            'success_rate': len(completed_tasks) / len(tasks) if tasks else 0,
            'task_results': self.task_results.copy(),
            'final_outputs': self._compile_final_outputs(tasks, completed_tasks)
        }
        
        # Update execution statistics
        self.execution_stats['total_tasks'] += len(tasks)
        self.execution_stats['completed_tasks'] += len(completed_tasks)
        self.execution_stats['failed_tasks'] += len(failed_tasks)
        
        if self.execution_stats['completed_tasks'] > 0:
            self.execution_stats['average_execution_time'] = (
                (self.execution_stats['average_execution_time'] * 
                 (self.execution_stats['completed_tasks'] - len(completed_tasks)) + 
                 execution_time) / self.execution_stats['completed_tasks']
            )
        
        logger.info(f"Task plan execution completed. Success rate: {results['success_rate']:.2%}")
        return results
    
    def _execute_single_task(self, task: AgentTask) -> TaskResult:
        """Execute a single task."""
        logger.debug(f"Executing task: {task.name}")
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.updated_at = datetime.now()
        
        try:
            # Add context from completed dependencies
            enhanced_context = task.context.copy()
            for dep_id in task.dependencies:
                if dep_id in self.task_results:
                    dep_result = self.task_results[dep_id]
                    if dep_result.status == TaskStatus.COMPLETED and dep_result.result:
                        enhanced_context.append({
                            'text': str(dep_result.result),
                            'source_task': dep_id,
                            'task_type': self.tasks[dep_id].task_type
                        })
            
            task.context = enhanced_context
            
            # Execute the task
            result = self.task_executor.execute_task(task)
            
            # Update task status
            task.status = result.status
            task.result = result
            task.updated_at = datetime.now()
            
            return result
            
        except Exception as e:
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
            task.status = TaskStatus.FAILED
            task.result = result
            task.updated_at = datetime.now()
            
            return result
    
    def _parse_plan_data(self, plan_data: Dict[str, Any], user_query: str, context: List[Dict[str, Any]]) -> List[AgentTask]:
        """Parse plan data into task objects."""
        tasks = []
        tasks_data = plan_data.get('tasks', [])
        
        for i, task_data in enumerate(tasks_data):
            task = AgentTask(
                task_id=str(uuid.uuid4()),
                name=task_data.get('name', f'Task {i+1}'),
                description=task_data.get('description', ''),
                task_type=task_data.get('type', 'analysis'),
                dependencies=task_data.get('dependencies', []),
                input_data=task_data.get('input', {}),
                context=context or []
            )
            tasks.append(task)
        
        return tasks
    
    def _create_default_plan(self, user_query: str, context: List[Dict[str, Any]] = None) -> List[AgentTask]:
        """Create a default task plan."""
        tasks = []
        
        # Retrieval task
        retrieval_task = AgentTask(
            task_id=str(uuid.uuid4()),
            name='Information Retrieval',
            description=f'Retrieve relevant information for: {user_query}',
            task_type='retrieval',
            input_data={'query': user_query, 'retrieval_type': 'comprehensive'},
            context=context or []
        )
        tasks.append(retrieval_task)
        
        # Analysis task
        analysis_task = AgentTask(
            task_id=str(uuid.uuid4()),
            name='Information Analysis',
            description=f'Analyze retrieved information for: {user_query}',
            task_type='analysis',
            dependencies=[retrieval_task.task_id],
            input_data={'prompt': user_query, 'analysis_type': 'detailed'},
            context=context or []
        )
        tasks.append(analysis_task)
        
        # Generation task
        generation_task = AgentTask(
            task_id=str(uuid.uuid4()),
            name='Content Generation',
            description=f'Generate comprehensive response for: {user_query}',
            task_type='generation',
            dependencies=[analysis_task.task_id],
            input_data={'prompt': user_query, 'generation_type': 'comprehensive'},
            context=context or []
        )
        tasks.append(generation_task)
        
        return tasks
    
    def _compile_final_outputs(self, tasks: List[AgentTask], completed_tasks: Set[str]) -> Dict[str, Any]:
        """Compile final outputs from completed tasks."""
        outputs = {
            'primary_output': None,
            'supporting_outputs': [],
            'summary': '',
            'next_steps': []
        }
        
        # Collect outputs from completed tasks
        for task in tasks:
            if task.task_id in completed_tasks and task.result and task.result.status == TaskStatus.COMPLETED:
                result_data = task.result.result
                
                if task.task_type == 'generation':
                    outputs['primary_output'] = result_data
                else:
                    outputs['supporting_outputs'].append({
                        'task_type': task.task_type,
                        'task_name': task.name,
                        'output': result_data
                    })
        
        # Generate summary using Gemini if we have outputs
        if outputs['supporting_outputs'] or outputs['primary_output']:
            try:
                summary_prompt = f"""
                Summarize the results of the following agent tasks for the user's query:
                
                Completed Tasks: {len(completed_tasks)}
                Primary Output: {outputs.get('primary_output', 'None')}
                Supporting Outputs: {len(outputs['supporting_outputs'])}
                
                Provide a concise summary highlighting key findings and recommendations.
                """
                
                response = self.task_executor.gemini_client.generate(summary_prompt)
                outputs['summary'] = response.text
            except:
                outputs['summary'] = f"Completed {len(completed_tasks)} tasks successfully."
        
        return outputs
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.execution_stats.copy()
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            return {
                'task_id': task_id,
                'name': task.name,
                'status': task.status.value,
                'progress': self._calculate_task_progress(task),
                'created_at': task.created_at.isoformat(),
                'updated_at': task.updated_at.isoformat()
            }
        return None
    
    def _calculate_task_progress(self, task: AgentTask) -> float:
        """Calculate task progress percentage."""
        if task.status == TaskStatus.COMPLETED:
            return 1.0
        elif task.status == TaskStatus.FAILED:
            return 0.0
        elif task.status == TaskStatus.RUNNING:
            # Rough estimate based on task type
            base_progress = 0.5
            if task.result:
                base_progress = 0.8
            return base_progress
        else:
            return 0.0